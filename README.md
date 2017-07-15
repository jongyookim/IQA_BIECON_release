# BIECON
A blind image evaluator based on a convolutional neural network.
This code implements the system described in the paper:

> J. Kim and S. Lee, “Fully deep blind image quality predictor,” IEEE Journal of Selected Topics in Signal Processing, vol. 11, no. 1, pp. 206–220, Feb. 2017.


## Prerequisites
This code was developed and tested with Theano 0.9.


## Environment setting
### Setting Database path:
For each database, set `BASE_PATH` in
`IQA_BIECON_release/data_load/LIVE`,
`IQA_BIECON_release/data_load/TID2008`,
`IQA_BIECON_release/data_load/TID2013`

### Setting local quality score map path:
Set `FR_MET_BASEPATH` and `FR_MET_SUBPATH_{DB}` in
`IQA_BIECON_release/data_load/data_loader_IQA`.

Detailed configuration is set in `NR_biecon.yaml` file.

- `fr_met`: This describes the name of the full-reference image quality assessment metric. ex) SSIM, FSIM ...
- `fr_met_scale`: This indicates the scale ratio of the local quality score maps to their original images.
- `fr_met_avg`: If True, for each divided image patch, local quality score maps are averaged to be scalar values. Otherwise, patch of local quality score maps are used.


## Running demo
We provide the demo code for running our model.
```bash
python example.py
```
