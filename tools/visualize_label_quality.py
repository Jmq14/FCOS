import numpy as np
import matplotlib.pyplot as plt

import os
import json
from tqdm import tqdm

import torch
from pycocotools.coco import COCO

from fcos_core.structures.bounding_box import BoxList
from fcos_core.structures.boxlist_ops import boxlist_iou


def convert_box_to_boxlist(boxes, image_width, image_height):
    boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
    boxes = BoxList(boxes, (image_width, image_height), mode="xywh").convert(
        "xyxy"
    )
    return boxes


if __name__ == "__main__":

    dataDir='datasets/coco'
    annFile_full='{}/annotations/instances_train2017_full.json'.format(dataDir)
    annFile='{}/annotations/instances_train2017_pseudo_stage2.json'.format(dataDir)
    coco_full = COCO(annFile_full)
    coco=COCO(annFile)

    feature_dir = '/home/mengqinj/capstone/output/stage1/coco_2017_train_partial/'
    predictions = torch.load(feature_dir + 'predictions.pth')

    image_ids=sorted(coco.getImgIds())
    catIds = list(range(2, 10))

    with open('results_back.json', 'r') as f:
        NNs = json.load(f)

    im_idx = -1
    i = 0
    for p in tqdm(predictions[:100]):
        if im_idx != -1:
            # plot last sequence
            # print(max_overlaps)
            # print(distances)
            # print(scores)
            # print('===')
            if len(max_overlaps)!= len(distances):
                print(len(max_overlaps), len(distances), len(scores))
            else:
                plt.scatter(max_overlaps, distances, c=scores, s=1, alpha=0.5)
        distances = []
        im_idx = image_ids[NNs[i]['image_id']]
        imginfo = coco.loadImgs(im_idx)[0]
        image_width = imginfo['width']
        image_height = imginfo['height']

        full_anns = coco_full.loadAnns(coco_full.getAnnIds(imgIds=(im_idx,), catIds=catIds))
        all_boxes = [obj["bbox"] for obj in full_anns]
        all_boxlist = convert_box_to_boxlist(all_boxes, image_width, image_height)

        query_box = p.resize((image_width, image_height))
        if len(query_box) == 0:
            i += 1
            im_idx = -1
            continue

        overlaps = boxlist_iou(query_box, all_boxlist).numpy()
        max_overlaps = overlaps.max(axis=1)

        labels = p.get_field('labels').tolist()
        scores = p.get_field('scores').tolist()

        while i < len(NNs) and im_idx == image_ids[NNs[i]['image_id']]:
            distances.append(NNs[i]['NN_distance'])
            i += 1
        
    
    plt.xlabel('IoU')
    plt.ylabel('NN distance')
    plt.colorbar()
    plt.clim(0, 1)
    plt.show()
