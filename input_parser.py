import os
import json
import re
import argparse

import pdb

from actor_critic import generate_retrospection, generate_action_proposals



def extract_messages_from_json(json_data, names):
    extracted_messages = []
    
    for entry in json_data:
        if "sender" in entry and entry["sender"] in names:
            name = entry["sender"]
            message = entry["message"]
            extracted_messages.append((name, message))
    
    return extracted_messages

def extract_feedback_from_json(json_data):
    feedback = []
    
    for entry in json_data:
        if "sender" in entry and entry["sender"] == "Feedback":
            feedback.append(entry["message"])
    
    return feedback

def parse_filename(filename):
    match = re.match(r'replan(\d+)_call(\d+)_agent(\w+)_', filename)
    if match:
        replan_number = int(match.group(1))
        call_number = int(match.group(2))
        agent_name = match.group(3)
        return replan_number, call_number, agent_name
    return None, None, None

def main(directory, step_number):
    # List to hold all messages in order
    conversation = []
    feedback_messages = []

    # Find the latest replan number
    replan_numbers = set()
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            replan_number, _, _ = parse_filename(filename)
            if replan_number is not None:
                replan_numbers.add(replan_number)
    
    if not replan_numbers:
        print("No replan files found.")
        return
    
    latest_replan = max(replan_numbers)
    print(f"Latest replan number detected: {latest_replan}")

    # Create a list to hold the filenames and their parsed data
    file_data = []

    # Iterate through the files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            parsed_replan, call_number, agent_name = parse_filename(filename)
            if parsed_replan == latest_replan and call_number is not None:
                file_data.append((call_number, filename))

    # Sort the files based on call number
    file_data.sort()

    # Process the sorted files
    for call_number, filename in file_data:
        with open(os.path.join(directory, filename), 'r') as f:
            json_data = json.load(f)
            messages = extract_messages_from_json(json_data, ["Alice", "Bob", "Chad"])
            conversation.extend(messages)
    
    # Prepare the output content
    output_content = []
    for i, (name, message) in enumerate(conversation):
        if i == 0 or conversation[i-1][0] != name:
            if i != 0:
                output_content.append("")  # Add a blank line between different agents
            output_content.append(f"{name}:")
        output_content.append(message)
    
    # Optionally, handle feedback files separately if needed
    feedback_pattern = f"replan{latest_replan}_feedback"
    for filename in os.listdir(directory):
        if filename.startswith(feedback_pattern) and filename.endswith(".json"):
            with open(os.path.join(directory, filename), 'r') as f:
                json_data = json.load(f)
                feedback = extract_feedback_from_json(json_data)
                feedback_messages.extend(feedback)

    if feedback_messages:
        output_content.append("")
        output_content.append("Feedback")
        output_content.append("[Environment Feedback]:")
        output_content.extend(feedback_messages)
    
    # Write the output to a text file
    output_filename = os.path.join(directory, f"replan{latest_replan}_summary.txt")
    with open(output_filename, 'w') as f:
        f.write("\n".join(output_content))
    
    print(f"Summary written to {output_filename}")

    retrospection = generate_retrospection("\n".join(output_content))
    retrospection_file = os.path.join(directory, f"replan{latest_replan}_retrospection.txt")
    with open(retrospection_file, 'w') as f:
        f.write(retrospection)
    print(f"Retrospection saved to {retrospection_file}")

    action_proposals = generate_action_proposals(retrospection_file)
    final_output_file = os.path.join(directory, f"replan{latest_replan}_final.txt")
    with open(final_output_file, 'w') as f:
        f.write("\n".join(output_content))
        f.write("\n\nRetrospection:\n")
        f.write(retrospection)
        f.write("\n\nAction proposer:\n")
        f.write(action_proposals)
    print(f"Final report with action proposals saved to {final_output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON files and generate summaries.")
    parser.add_argument("-run", type=int, required=True, help="Run number")
    parser.add_argument("-step", type=int, required=True, help="Step number")
    
    args = parser.parse_args()

    # Construct the directory path
    directory_path = f"../data/test/run_{args.run}/step_{args.step}/prompts"
    
    # Run the main function with the provided directory and step number
    main(directory_path, args.step)
