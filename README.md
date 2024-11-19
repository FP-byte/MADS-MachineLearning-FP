# MADS-MachineLearning-FP

# Installation
The installation guide assumes a UNIX system (os x or linux).
If you have the option to use a VM, see the references folder for lab setups (both for azure and surf).
For the people that are stuck on a windows machine, please use [git bash](https://gitforwindows.org/) whenever I 
refer to a terminal or cli (command line interface).

## install rye as package manager
please note that rye might already be installed on your machine.
1. watch the [introduction video about rye](https://rye.astral.sh/guide/)
2. You skipped the video, right? Now go back to 1. and actually watch it. I'll wait.
3. check if rye is already installed with `which rye`. If it returns a location, eg `/Users/user/.rye/shims/rye`, rye is installed.
4. else, install [rye](https://rye.astral.sh/) with `curl -sSf https://rye.astral.sh/get | bash`

run through the rye installer like this:
- platform linux: yes
- preferred package installer: uv
- Run a Python installed and managed by Rye
- which version of python should be used as default: 3.11
- should the installer add Rye to PATH via .profile? : y
- run in the cli: `source "$HOME/.rye/env"`

For windows this should be the same, except for the platform off course...

## install dependencies

To install the project dependencies:
`rye sync`

## Install CUDA for GPU 
For Windows: install NVIDIA drivers and Nvidia tools
Then add the following to your pyproject.toml:
```
[[tool.rye.sources]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu118"
```
This should download the CUDA version of pytorch the first time you sync.

#Now add torch to your setup
`rye add pytorch torchvision`

## Start your environment
In windows:
`source .venv/Scripts/activate`
In Linux or Mac:
`source .venv/activate`

