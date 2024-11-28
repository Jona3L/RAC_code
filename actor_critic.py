import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from huggingface_hub import login

# Hugging Face login
hf_token = "your hf token here"
login(hf_token)

# Initialize Llama 3.1 model
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# Path planning instruction (unchanged)
PATH_PLAN_INSTRUCTION = """
[Path Plan Instruction]
Each <coord> is a tuple (x,y,z) for gripper location, follow these steps to plan:
1) Decide target location (e.g. an object you want to pick), and your current gripper location.
2) Plan a list of <coord> that move smoothly from current gripper to the target location.
3) The <coord>s must be evenly spaced between start and target.
4) Each <coord> must not collide with other robots, and must stay away from table and objects.  
[How to Incorporate [Environment Feedback] to improve plan]
    If IK fails, propose more feasible step for the gripper to reach. 
    If detected collision, move robot so the gripper and the inhand object stay away from the collided objects. 
    If collision is detected at a Goal Step, choose a different action.
    To make a path more evenly spaced, make distance between pair-wise steps similar.
        e.g. given path [(0.1, 0.2, 0.3), (0.2, 0.2. 0.3), (0.3, 0.4. 0.7)], the distance between steps (0.1, 0.2, 0.3)-(0.2, 0.2. 0.3) is too low, and between (0.2, 0.2. 0.3)-(0.3, 0.4. 0.7) is too high. You can change the path to [(0.1, 0.2, 0.3), (0.15, 0.3. 0.5), (0.3, 0.4. 0.7)] 
    If a plan failed to execute, re-plan to choose more feasible steps in each PATH, or choose different actions.
"""

def generate_text(prompt, max_tokens=300, temperature=0.7):
    messages = [
        {"role": "system", "content": "You are here to check why the above situation has failed. Please provide a 100 words retrospective on why it failed when interacting with the environment. No need to do recommendation:"},
        {"role": "user", "content": prompt},
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    outputs = pipe(
        input_text,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
    )

    # Clean up the output
    raw_output = outputs[0]['generated_text']
    # Remove everything up to and including <|end_header_id|>
    cleaned_output = raw_output.split("<|end_header_id|>")[-1]
    # Remove <|eot_id|> if present
    cleaned_output = cleaned_output.replace("<|eot_id|>", "")
    # Trim any leading or trailing whitespace
    cleaned_output = cleaned_output.strip()
    
    return cleaned_output
    
    #return outputs[0]['generated_text'].split(prompt)[-1].strip()

def generate_retrospection(prompt, max_tokens=200, temperature=0.7):
    combined_prompt = PATH_PLAN_INSTRUCTION + "\n\n" + prompt
    print("========")
    print(prompt)
    return generate_text(combined_prompt, max_tokens, temperature)

def generate_action_proposals(retrospection_file, max_tokens=200, temperature=0.7):
    with open(retrospection_file, "r") as file:
        retrospection_content = file.read()

    action_prompt = f"Based on the following retrospection and previous context conversation, propose specific solution only for parameter change within 100 words:\n\n{retrospection_content}"

    messages = [
        {"role": "system", "content": "You are the supervisor of the action proposer. Propose adjustments based on the retrospection provided."},
        {"role": "user", "content": action_prompt},
    ]
    
    return generate_text(action_prompt, max_tokens, temperature)

def main(input_file):
    with open(input_file, "r") as file:
        input_text = file.read()

    prompt = f"""{input_text}"""
    try:
        retrospection = generate_retrospection(prompt)
    except Exception as e:
        print(f"Error generating retrospection: {e}")
        return

    # Define the retrospection file name
    retrospection_file = os.path.splitext(input_file)[0] + "_retrospection.txt"

    # Write the input and retrospection to the retrospection file
    with open(retrospection_file, "w") as out_file:
        out_file.write("Input:\n")
        out_file.write(input_text + "\n\n")
        out_file.write("Retrospection:\n")
        out_file.write(retrospection)

    print(f"Retrospection saved to {retrospection_file}")

    # Generate action proposals based on the retrospection
    action_proposals = generate_action_proposals(retrospection_file)
    
    # Define the final output file name
    final_output_file = os.path.splitext(input_file)[0] + "_final.txt"

    # Write all previous information and the action proposals to the final output file
    with open(final_output_file, "w") as out_file:
        with open(retrospection_file, "r") as ret_file:
            out_file.write(ret_file.read())
        out_file.write("\n\nAction proposer:\n")
        out_file.write(action_proposals)

    print(f"Final report with action proposals saved to {final_output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to the input text file")
    args = parser.parse_args()

    main(args.input_file)