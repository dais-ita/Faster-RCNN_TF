# Faster-RCNN_TF

This is an experimental Tensorflow implementation of Faster RCNN - a convnet for object detection with a region proposal network.
For details about R-CNN please refer to the paper [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/pdf/1506.01497v3.pdf) by Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.


### Requirements: software

1. Requirements for Tensorflow (see: [Tensorflow](https://www.tensorflow.org/))

2. Python packages you might not have: `cython`, `python-opencv`, `easydict`

For installing `OpenCV` I followed this [tutorial](http://stackoverflow.com/questions/37188623/ubuntu-how-to-install-opencv-for-python3).
or if using Anaconda, you can use
```
conda install opencv
```
### Requirements: hardware

1. For training the end-to-end version of Faster R-CNN with VGG16, 3G of GPU memory is sufficient (using CUDNN)

### Installation (sufficient for the demo)

1. Clone the repository
  ```Shell
  # Make sure to clone with --recursive
  git clone --recursive https://github.com/smallcorgi/Faster-RCNN_TF.git
  ```
2. Move to the `dais` branch.
```Shell
git checkout dais
```

2. Build the Cython modules
    ```Shell
    cd $FRCN_ROOT/lib
    make
    ```

### Demo

*After successfully completing [basic installation](#installation-sufficient-for-the-demo)*, you'll be ready to run the demo.

Download model training on PASCAL VOC 2007  [[Google Drive]](https://drive.google.com/open?id=0ByuDEGFYmWsbZ0EzeUlHcGFIVWM) [[Dropbox]](https://www.dropbox.com/s/cfz3blmtmwj6bdh/VGGnet_fast_rcnn_iter_70000.ckpt?dl=0)

To run the demo
```Shell
cd $FRCN_ROOT
python ./tools/demo.py --model model_path
```
The demo performs detection using a VGG16 network trained for detection on PASCAL VOC 2007.



###References
[Faster R-CNN caffe version](https://github.com/rbgirshick/py-faster-rcnn)

[A tensorflow implementation of SubCNN (working progress)](https://github.com/yuxng/SubCNN_TF)

