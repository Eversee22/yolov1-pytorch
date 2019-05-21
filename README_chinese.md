## yolov1-pytorch
YOLOv1的pytorch版本，但使用了不同的主干网络, 损失函数的计算上也有所差异，不过取得了比较不错的结果:

| model(Train on voc2012+2007)| backbone            | map@voc2007test | FPS(Geforce GTX 1070)  |
| --------------------        | --------------      | ----------      | -------   |
| YOLO_ResNet50_7x7           |   ResNet50          | 66.8%           |  46   |
| YOLO_ResNet50_14x14         |   ResNet50 + DetNet | 69.5%           |  35   |
| YOLO_VGG16_7x7              |   VGG16             | 61.8%           |  53   |

## 依赖包(Python3)
- PyTorch-1.0
- opencv
- tqdm
- visdom(optional)

## 使用
首先你需要下载并转换到当前分支:
git clone https://https://github.com/Eversee22/yolov1-pytorch/
cd yolov1-pytorch
git checkout exp1-loss



