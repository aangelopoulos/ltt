# Learn then Test

This repository will allow you to reproduce the experiments in the Learn then Test paper.
Soon, I will add a colab notebook to this repository as well.
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
to add the following on line 68:
```
results.roi_masks = roi_masks
```

Then the experiments should run.
