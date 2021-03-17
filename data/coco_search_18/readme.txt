Introduction:

COCO-Search18 is a laboratory-quality dataset of goal-directed behavior large enough to train deep-network models. It consists of the eye gaze behavior from 10 people searching for each of 18 target-object categories in 6202 natural-scene images, yielding ~300,000 search fixations. The training, validation, and test images in COCO-Search18 are already freely available as part of COCO. Researchers are also free to see and use COCO-Search18’s training and validation search fixations, but the fixations on the test images are withheld. As part of a separate benchmark track, it will be possible to upload predictions and have then evaluated on the test dataset. We hope you enjoy using COCO-Search18.

If you use COCO-Search18, please cite:

Yang, Z., Huang, L., Chen, Y., Wei, Z., Ahn, S., Zelinsky, G., Samaras, D., & Hoai, M. (2020). Predicting Goal-directed Human Attention Using Inverse Reinforcement Learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 193-202).

@InProceedings{Yang_2020_CVPR,
author = {Yang, Zhibo and Huang, Lihan and Chen, Yupei and Wei, Zijun and Ahn, Seoyoung and Zelinsky, Gregory and Samaras, Dimitris and Hoai, Minh},
title = {Predicting Goal-Directed Human Attention Using Inverse Reinforcement Learning},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}


Data format:

Fixation data is saved in .json format.
*split1 files have the same train/valid split as was done in the paper.
*split2 files have all training images from COCO2014 traning set and all validation images from COCO2014 validation set. 
Testing images are all from COCO2014 validation set and are the same for both split1 and split2.
All images used in data collection were resized to 1680x1050 with zero padding and aspect ratio kept. 
Images can be downloaded at http://vision.cs.stonybrook.edu/~cvlab_download/COCOSearch18-images-TP.zip.


'name': image name,
'subject': subject number,
'task': target object category,
'condition': target present or target absent images,
'bbox': bounding box coordinates of the target [x,y,w,h],
'X': X coordinates of fixations from the first fixation to the last in this trial (including the initial center fixation),
'Y': Y coordinates of fixations from the first fixation to the last in this trial (including the initial center fixation),
'T': fixation durations from the first fixation to the last in this trial (including the initial center fixation),
'length': number of total fixations in this trial,
'correct': correct response = 1, incorrect resonse = 0,
'RT': search reaction time,
'split': training set = 'train', validation set = 'valid'


---------------v1.0----------------------  Date: 06/2020
In this initial stage of release, only fixations made on target-present search trials are available at this time.

