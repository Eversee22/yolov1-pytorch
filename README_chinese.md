## yolov1-pytorch
YOLOv1的pytorch版本，但使用了不同的主干网络, 损失函数的计算上也有所差异，不过取得了比较不错的结果:
| model(Train on voc2012+2007)                | backbone | map@voc2007test  | FPS(Geforce GTX 1070)  |
| -------------------- | -------------- | ---------- | -------   |
| YOLO_ResNet50_7x7  |   ResNet50        | 66.8%      |  46   |
| YOLO_ResNet50_14x14  |   ResNet50 + DetNet        | 69.5%      |  35   |
| YOLO VGG-16_7x7  |   VGG-16        | 61.8%      |  53   |

