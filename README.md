# PixelLink-with-pytorch
PixelLink-with-pytorch

## Prerequisite

- python-3.6
- pytorch-0.4.0
- torchvision-0.2.1
- opencv-3.4.0.14
- numpy-1.14.3
- Pillow-5.5.0

They could all be installed through pip except pytorch and torchvision. As for pytorch and torchvision, they both depends on your CUDA version, you would prefer to reading [pytorch's official site](https://pytorch.org/)

## Structure
All main source code is in the root directory of the project. 
- ${project_root}/unittest contains code you could run indenpendently, which identifies some modules's function.
- datasets.py contains code which generates datasets and preprocess code
- net.py contains the neural network structure
- criterion.py contains code which calculates the loss
- postprocess.py contains code for data postprocessing, which transform pixel and link mask to bounding boxes
- config.py contains almost all changeable parameters.
- other *.py are useless, they exists only for re-constructing the project later on.


## Train
### Before starting
You could modify training parameters in ${project_root}/config.py
You need to download the dataset [here](http://rrc.cvc.uab.es/?ch=4&com=tasks) and unzip it as ${project_root}/train_images/images and ${project_root}/train_images/ground_truth
`python main.py --train 1`

## Retrain
Be sure there is a pretrained model in the ${project_root}/models directory
`python main.py --train 1 --retrain 1`

## Test
Be sure there is a pretrained model in the ${project_root}/models directory
`python main.py`

## Noted

There are still some bugs in source code. The result is not satisfactory. Still under developing...
