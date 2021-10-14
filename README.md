# Learn then Test

[If you see this note, congratulations---you're here early!  I haven't yet publicized this paper, but I needed the repo to be public :) Nonetheless, you should be able to run the code if you really want to!]

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
to add the following on line 67 (see the file experiments/detection/postprocessing.py):
```
results.roi_masks = roi_masks
```

Then the experiments should run.
