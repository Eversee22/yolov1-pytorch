!wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
!wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
!tar xf VOCtrainval_06-Nov-2007.tar
!tar xf VOCtest_06-Nov-2007.tar
!git clone https://github.com/Eversee22/yolov1-pytorch.git
!cd yolov1-pytorch
!git checkout exp1-loss
!cd ../
!mv yolov1-pytorch/*.* .
!mv yolov1-pytorch/mmodels .
!mv yolov1-pytorch/cfg .
!mv yolov1-pytorch/data .
#from google.colab import files
#files.upload()
!python voc_label_07test.py
!wget 104.129.181.45/file/resnet50_final.pth
!python detect.py -i data/2007_test.txt -t eavl resnet50_final.pth
!python reval_voc_py3.py results