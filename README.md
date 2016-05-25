# Histogram of oriented gradients GPU
Implementation of HOG (histogram of oriented gradients) on GPU using OpenCL

## Example
```bash
$ ./hog input [device] [load] [train] [visual]
```
* `input`:  use `--input 'path_to_image'` or `--video 'path_to_video'`      
* `device`: specify device as `--device GPU` or `--device CPU` (CPU is implicit)
* `load`:   load linear SVM model `--load 'path_to_model'`
* `train`:  train new model `--train 'path_to_positive' 'path_to_negative'`
* `visual`: visualize HOG features `--visual`


## Usage

## Licence
GNU LGPL v3 (see LICENCE)
