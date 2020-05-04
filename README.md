## yolov1-pytorch
Implementation based on YOLOv1 with pytorch, though some differences in backbone network and loss calculation existing. And I get somewhat not bad results:

| model(Train on voc2012+2007)|  backbone          | map@voc2007test | FPS(Geforce GTX 1070)  |
| --------------------        |  ------------      | ----------      | -------   |
| YOLO                        |  ——                | 63.4% (Paper)   |  45 (Paper)   |
| YOLO_ResNet50_7x7           |  ResNet50          | 66.8%           |  46   |
| YOLO_ResNet50_14x14         |  ResNet50 + [DetNet](https://arxiv.org/abs/1804.06215) | 69.5%           |  35   |
| YOLO_VGG16_7x7              |  VGG16_bn             | 61.8%           |  53   |

## Requirements (python3)
- pytorch-1.0
- opencv
- tqdm
- visdom(optional)

## Download:  
  ```
  git clone https://github.com/Eversee22/yolov1-pytorch.git  
  ```

## Detection  

  * Image

    `python detect.py -i data/person.jpg weights.pth`

    The above code will detect a single picture, where `weights.pth` needs to be replaced with the actual weight path. The result of the above execution are as follows:

    ![](det/bbox_person.png)

    You can also detect all images (png or jpg) in a directory:  

    `python detect.py -i data -t test2 weights.pth `

    Or enter a file including image paths:  

    `python detect.py -i 2007_test.txt -t test2 weights.pth`

  * Video

    `python video.py -i yourvideofile weights.pth`

    The above command will display the real-time video detection result. However, depending on the hardware, the detection frame rate may be different. And if you want to save the detected video at the frame rate of the original video, use the following command:  

    `python video.py -i yourvideofile --dv 1 weights.pth`

    It will save the video in avi format in the `local` directory.

  **Note: By default, the model of ResNet50_14x14 is used. If you use the 7x7 model, you need to change the `yolond` file in the `cfg` directory, and change `side` to 7, `downsm` to 1 and `det` to 0.**

## Training  
  Download the VOC 2007 and VOC 2012 datasets, refer to the following links：  
  ```
  https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
  https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
  https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
  ```
  Unpack above data and the `VOCdevkit` directory should appear where VOC2007 and VOC2012 exist. Then run `python voc_label_bbox.py` to generate training files. As expected, there will be 4 files in the `data` directory: `voc_2007_train.txt voc_2007_val.txt voc_2012_train.txt voc_2012_val.txt`. What you need to do next is to merge the 4 files to one training file. For example, you can do that by running `cat voc_2007_train.txt voc_2007_val.txt voc_2012_* > train.txt` in Linux.

  **Note: Depending on the location of `VOCdevkit`, you may need to change the `VOC_root` parameter in `voc_label_bbox.py` to specify the path where `VOCdevkit` is located. By default the `data` directory is assigned.**

  Once the data is ready, run the command `python train.py` to start training. If there are no errors in the configuration, the training process will proceed normally. The default training epoch is 50 and it takes around 3 to 4 hours when training on VOC 2007 and using single GTX 1070 GPU . In VOC 2007+2012, it is roughly three times longer. Although training on CPU is allowed, but it is not recommended, unless the time is not your problem. If you want to visualize the training process, you need to pre-install the `visdom` package and use `python -m visdom.server` to enable the visdom service firstly. Then set an environment parameter when using the training command, such as `python train.py -env test `, it will open the visdom visual environment named `test` and you can observe the loss curve of the training in your browser then.

  **Note: ResNet50 is used as the backbone network by default. If you use VGG16, you need to assign the `-n` parameter, such as `python train.py -n vgg16`. You also need to pay attention to this when detecting, otherwise the weight loading process may encounter erros. In addition, for ResNet50, you can use 7x7 or 14x14 output, which is consistent with the configuration at the time of detection.**

## mAP's calculation
  run `python voc_label_07test.py` to get test data, then run the following commands:  
  ```
  python detect.py -i data/2007_test.txt -t eval weights.pth
  python reval_voc_py3.py results
  ```
### [Chinese](README_chinese.md)
