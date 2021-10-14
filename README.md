# Learn then Test
[Please e-mail me if you have trouble with this repository!]

To install the dependencies for our experiments, run the following command from the root directory of the repository:
```
conda env create -f environment.yml
conda activate ltt
```

Then you should be able to run the experiment scripts.

The detectron code is different, and requires a separate set of dependencies.
You will also need to make some modifications to the detectron source code.

```
vim ~/anaconda3/envs/detectron2/lib/python3.8/site-packages/detectron2/modeling/postprocessing.py
```

Then on line 68, add
```
results.roi_masks = roi_masks
```
