# Integrating Retrospective Actor-Critic Framework in Multi-Robot
Collaboration based on Large Language Model
Codebase for paper: Integrating Retrospective Actor-Critic Framework in Multi-Robot
Collaboration based on Large Language Model


## Setup
### setup conda env and package install
```
conda create -n RAC python=3.8 
conda activate RAC
```
### Install mujoco and dm_control 
```
pip install mujoco==2.3.0
pip install dm_control==1.0.8 
```
### Install other packages
```
conda env update --name RAC --file environment.yaml

```

### Notes on swithing the access token 
in files named actor_critic.py, dialog_prompter.py, and plan_prompter, access token for huggingface is required

### Example Usage(sweep for example) 
$ conda activate RAC

module load mesa/20.2.1

export LIBGL_ALWAYS_SOFTWARE=1

export MUJOCO_GL=osmesa

(RAC) $ python run_dialog.py --task sweep --skip_display


