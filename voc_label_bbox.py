import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

VOC_root = 'data/'

# def convert(size, box):
#     dw = 1./size[0]
#     dh = 1./size[1]
#     x = (box[0] + box[1])/2.0
#     y = (box[2] + box[3])/2.0
#     w = box[1] - box[0]
#     h = box[3] - box[2]
#     x = x*dw
#     w = w*dw
#     y = y*dh
#     h = h*dh
#     return (x,y,w,h)


def convert_annotation(year, image_id):
    in_file = open(VOC_root+'VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    #out_file = open('VOCdevkit/VOC%s/labels/%s.txt'%(year, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    bbs=[]
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text), cls_id)
        #bb = convert((w,h), b)
        #out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        bbs.append(b)

    return bbs

# wd = getcwd()
jn = False

for year, image_set in sets:
    #if not os.path.exists('VOCdevkit/VOC%s/labels/'%(year)):
    #    os.makedirs('VOCdevkit/VOC%s/labels/'%(year))
    image_ids = open(VOC_root+'VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('data/voc_%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        bbs = convert_annotation(year,image_id)
        if jn:
            list_file.write('%s.jpg ' % (image_id))
        else:
            list_file.write(VOC_root+'VOCdevkit/VOC%s/JPEGImages/%s.jpg '%(year, image_id))
        for b in bbs:
            list_file.write(" ".join([str(a) for a in b])+" ")
        list_file.write('\n')
        #convert_annotation(year, image_id)
    list_file.close()
