# BIECON
A blind image evaluator based on a convolutional neural network (BIECON) is a no-reference image quality method using a CNN.
This code implements the system described in the following paper:

> J. Kim and S. Lee, “Fully deep blind image quality predictor,” IEEE Journal of Selected Topics in Signal Processing, vol. 11, no. 1, pp. 206–220, Feb. 2017.

## Prerequisites
This code was developed and tested with Theano 0.9, CUDA 8.0, and Windows.

## Generating local quality score maps
Set `BASE_PATH` to the actual root path of each database.
Set `FR_MET_BASEPATH` and `FR_MET_SUBPATH` in `gen_local_metric_scores.m`.
For each database, data will be stored in "`FR_MET_BASEPATH` + `FR_MET_SUBPATH`".
Then run `gen_local_metric_scores.m` using Matlab. We provide a SSIM metric as default.

## Environment setting
### Setting database path:
For each database, set `BASE_PATH` to the actual root path of each database in the following files:
`IQA_BIECON_release/data_load/LIVE`,
`IQA_BIECON_release/data_load/TID2008`, and
`IQA_BIECON_release/data_load/TID2013`.

(These `BASE_PATH` should be same to the `BASE_PATH` in `gen_local_metric_scores.m`.)

### Setting local quality score map path:
Set `FR_MET_BASEPATH` and `FR_MET_SUBPATH_{DB name}` in
`IQA_BIECON_release/data_load/data_loader_IQA`.
{*DB name*} can be *LIVE*, *TID2008*, or *TID2013*

(These should be same to those in `gen_local_metric_scores.m`.)

Detailed configuration of local quality score maps is set in `NR_biecon.yaml`.
- `fr_met`: This describes the name of the full-reference image quality assessment metric. The corresponding local quality score maps must be generated first. ex) SSIM, FSIM ...
- `fr_met_scale`: This indicates the scale ratio of the local quality score maps to their original images.
- `fr_met_avg`: If True, for each divided image patch, local quality score maps are averaged to be scalar values. Otherwise, patch of local quality score maps are used.


## Training BIECON
We provide the demo code for training a BIECON model.
```bash
python example.py
```

- `tr_te_file`: Store the randomly divided (training and testing) reference image indices in this file.
- `snap_path`: This indicates the path to store snapshot files
