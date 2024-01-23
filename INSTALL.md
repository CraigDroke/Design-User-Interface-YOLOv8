# Installation Guide
This is being built with python 3.11.7. Try to use that version if possible.
## Guide for this Project
We are currently working on setting up the correct requirements.txt file for the entire interface. This will be updated soon!
### Current Method
1. Create a conda environment.
    - conda create -n yolo8inter python=3.11.7
2. Activate conda environment.
    - conda activate yolo8inter
3. Install PyTorch. The code below is specified for Windows, Conda, Python, & CUDA 11.8. If you need a different build, go to the [PyTorch website](https://pytorch.org/get-started/locally/).
    - conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
4. Install Ultralytics.
    - pip install ultralytics==8.0.186
5. Install Other Requirements, like GRADIO. This is implemented but will be continually updated.
    - pip install -r requirements.txt
### Other Recommendations
1. When you run any of the examples, at least 1 weights file (.pt) will be downloaded. It is reccomended to create a "weights" folder in the directory to keep track of these.

## Directly from Ultralytics Github
See below for a quickstart installation and usage example, and see the [YOLOv8 Docs](https://docs.ultralytics.com) for full documentation on training, validation, prediction and deployment.

<details open>
<summary>Install</summary>

Pip install the ultralytics package including all [requirements](https://github.com/ultralytics/ultralytics/blob/main/pyproject.toml) in a [**Python>=3.8**](https://www.python.org/) environment with [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).

[![PyPI version](https://badge.fury.io/py/ultralytics.svg)](https://badge.fury.io/py/ultralytics) [![Downloads](https://static.pepy.tech/badge/ultralytics)](https://pepy.tech/project/ultralytics)

```bash
pip install ultralytics
```
### Test 
For alternative installation methods including [Conda](https://anaconda.org/conda-forge/ultralytics), [Docker](https://hub.docker.com/r/ultralytics/ultralytics), and Git, please refer to the [Quickstart Guide](https://docs.ultralytics.com/quickstart).

</details>
