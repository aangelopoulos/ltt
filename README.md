# Learn then Test
<p align="center">
    <a style="text-decoration:none !important;" href="https://arxiv.org/abs/2110.01052" alt="arXiv"> <img src="https://img.shields.io/badge/paper-arXiv-red" /> </a>
    <a style="text-decoration:none !important;" href="https://docs.conda.io/en/latest/miniconda.html" alt="package management"> <img src="https://img.shields.io/badge/conda-env-green" /> </a>
    <a style="text-decoration:none !important;" href="https://opensource.org/licenses/MIT" alt="License"> <img src="https://img.shields.io/badge/license-MIT-blue.svg" /> </a>
    <a href="https://colab.research.google.com/github/aangelopoulos/ltt/blob/main/Detectron2%2BLTT.ipynb" target="_parent"><img src="https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg"></a>

</p>

This repository will allow you to reproduce the experiments in the Learn then Test paper.
For now, please e-mail me if you have trouble reproducing our results.

To install the dependencies for our experiments, run the following command from the root directory of the repository:
```
conda env create -f environment.yml
conda activate ltt
```

Then you should be able to run the experiment scripts.

The detectron code is different, and requires a separate set of dependencies.
You will also need to make some modifications to the detectron source code.

In the /experiments/detectron folder, please execute
```
conda env create -f environment.yml
conda activate detectron2
```
Then you will need to edit the file 
```
~/anaconda3/envs/detectron2/lib/python3.8/site-packages/detectron2/modeling/postprocessing.py
```
to add the following on line 67 (see the file experiments/detection/postprocessing.py):
```
results.roi_masks = roi_masks
```

Then the experiments should run.
