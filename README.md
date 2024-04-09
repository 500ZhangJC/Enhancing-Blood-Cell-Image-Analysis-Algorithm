# Enhancing-Blood-Cell-Image-Analysis-Algorithm
IGGM-YOLO is a deep learning algorithm designed for accurate blood cell image classification. Leveraging intelligent perception optimization and multi-path residual feature fusion, the algorithm aims to address challenges such as small datasets, label inaccuracies, and sample imbalances. With IGGM-YOLO, you can enhance feature representation of blood cell images, mitigate label inaccuracies, and improve recognition of blood cell categories.

1. Dataset Preparation
Before training, place the label files under the 'Annotation' folder in the 'VOC2007' folder within the 'VOCdevkit' directory.
Before training, place the image files under the 'JPEGImages' folder in the 'VOC2007' folder within the 'VOCdevkit' directory.

2. Dataset Processing
After arranging the dataset, we need to use voc_annotation.py to obtain the training files '2007_train.txt' and '2007_val.txt'.
Modify the parameters in voc_annotation.py. For the first training, you can simply modify 'classes_path', which points to the txt file containing the detection categories.
When training your own dataset, you can create a 'cls_classes.txt' containing the categories you need to distinguish.
The contents of the 'model_data/cls_classes.txt' file are:
```python
cat
dog
...
```
3. Start Network Training
There are many parameters for training, all of which are in train.py. Carefully read the comments after downloading the library, and the most important part is still 'classes_path' in train.py.
'classes_path' points to the txt file corresponding to the detection categories, which is the same as the txt file in voc_annotation.py.
After modifying 'classes_path', you can run train.py to start training. After training for multiple epochs, the weights will be generated in the 'logs' folder.

4. Predicting Training Results
Predicting training results requires two files, yolo.py and predict.py. Modify 'model_path' and 'classes_path' in yolo.py.
'model_path' points to the trained weight file in the 'logs' folder.
'classes_path' points to the txt file corresponding to the detection categories.
After completing the modification, you can run predict.py for detection. Enter the image path after running to start detection.

## Prediction Steps
### a. Using Trained Weights
Train according to the training steps.
In the yolo.py file, modify the 'model_path' and 'classes_path' in the following section to correspond to the trained files; 'model_path' corresponds to the weight file in the 'logs' folder, and 'classes_path' is the classes corresponding to 'model_path'.

```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   When using your own trained model for prediction, be sure to modify model_path and classes_path!
    #   model_path points to the weight file under the logs folder, and classes_path points to the txt file under model_data
    #
    #   After training, there are multiple weight files under the logs folder. Choose the one with the lowest validation set loss.
    #   Lower validation set loss does not necessarily mean higher mAP, it only means that the weight performs better on the validation set.
    #   If there is a shape mismatch, pay attention to modifying the model_path and classes_path parameters during training.
    #--------------------------------------------------------------------------#
    "model_path"        : 'model_data/yolov7_weights.pth',
    "classes_path"      : 'model_data/coco_classes.txt',
    #---------------------------------------------------------------------#
    #   anchors_path represents the txt file corresponding to the prior box, generally do not modify.
    #   anchors_mask is used to help the code find the corresponding prior box, generally do not modify.
    #---------------------------------------------------------------------#
    "anchors_path"      : 'model_data/yolo_anchors.txt',
    "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
    #---------------------------------------------------------------------#
    #   The size of the input image, must be a multiple of 32.
    #---------------------------------------------------------------------#
    "input_shape"       : [640, 640],
    #------------------------------------------------------#
    #   The version of yolov7 used, this repository provides two:
    #   l : corresponds to yolov7
    #   x : corresponds to yolov7_x
    #------------------------------------------------------#
    "phi"               : 'l',
    #---------------------------------------------------------------------#
    #   Only prediction boxes with scores greater than confidence will be retained
    #---------------------------------------------------------------------#
    "confidence"        : 0.5,
    #---------------------------------------------------------------------#
    #   The nms_iou used for non-maximum suppression
    #---------------------------------------------------------------------#
    "nms_iou"           : 0.3,
    #---------------------------------------------------------------------#
    #   This variable is used to control whether to use letterbox_image to resize the input image without distortion,
    #   After multiple tests, it was found that closing letterbox_image and directly resizing performed better
    #---------------------------------------------------------------------#
    "letterbox_image"   : True,
    #-------------------------------#
    #   Whether to use Cuda
    #   Set to False if there is no GPU
    #-------------------------------#
    "cuda"              : True,
}
```
3. Running predict.py

## Evaluation Steps
### a. Evaluating the BCCD Test Set
1. This paper uses the VOC format for evaluation.
2. Modify 'model_path' and 'classes_path' in yolo.py. 'model_path' points to the trained weight file in the logs folder. 'classes_path' points to the txt file corresponding to the detection categories.
3. Running get_map.py will generate evaluation results, and the results will be saved in the map_out folder.
