import os
import numpy as np
from pycocotools.coco import COCO
import cv2
from tqdm import tqdm
import argparse
import json

import torch

np.random.seed(0)

def convert_to_xyxy(bbox):
    top_left = (int(bbox[0]), int(bbox[1]))
    bottom_right = (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3]))
    return top_left, bottom_right

if __name__ == "__main__":
    dataDir = 'datasets/coco'
    dataType = 'train2017'
    image_dir = dataDir + '/' + dataType + '/'
    output_dir = 'output_images/'
    annFile_pseudo = '{}/annotations/instances_{}_pseudo.json'.format(dataDir,dataType)
    annFile_full = '{}/annotations/instances_{}_full.json'.format(dataDir,dataType)
    coco_pseudo =COCO(annFile_pseudo)
    coco_full = COCO(annFile_full)

    sample_num = 100

    img_ids = np.random.choice(sorted(coco_pseudo.getImgIds()), sample_num, replace=False)
    print(img_ids)
    catIds = list(range(2, 10))

    for img_id in img_ids:
        imginfo = coco_pseudo.loadImgs([img_id,])[0]
        img = cv2.imread(image_dir + imginfo['file_name'])

        pseudo_anns = coco_pseudo.loadAnns(coco_pseudo.getAnnIds(imgIds=(img_id,)))
        full_anns = coco_full.loadAnns(coco_full.getAnnIds(imgIds=(img_id,), catIds=catIds))

        pseudo_boxes = [obj["bbox"] for obj in pseudo_anns if "ispseudo" in obj.keys()]
        partial_boxes = [obj["bbox"] for obj in pseudo_anns if "ispseudo" not in obj.keys()]
        full_boxes = [obj["bbox"] for obj in full_anns]

        # print(len(pseudo_boxes), len(partial_boxes), len(full_boxes))

        for box in full_boxes:
            # blue
            top_left, bottom_right = convert_to_xyxy(box)
            img = cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 2)

        for box in partial_boxes:
            # red
            top_left, bottom_right = convert_to_xyxy(box)
            img = cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)
        
        for box in pseudo_boxes:
            # yellow
            top_left, bottom_right = convert_to_xyxy(box)
            img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 255), 2)

        cv2.imwrite(output_dir+imginfo['file_name'], img)