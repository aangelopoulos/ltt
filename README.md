# Learn then Test
[Please e-mail me if you have trouble with this repository!]

For the detectron code, after pip install detectron, execute
```
vim ~/anaconda3/envs/detectron2/lib/python3.8/site-packages/detectron2/modeling/postprocessing.py
```

Then on line 68, add
```
results.roi_masks = roi_masks
```
