import os
import numpy as np
from pycocotools.coco import COCO
import cv2
from tqdm import tqdm
import argparse
import json

import torch

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
    dataType='train2017'
    annFile='{}/annotations/instances_{}_pseudo.json'.format(dataDir,dataType)
    annFile_ful='{}/annotations/instances_{}_full.json'.format(dataDir,dataType)
    coco=COCO(annFile)
    coco_full = COCO(annFile_ful)

    image_ids=sorted(coco.getImgIds())
    catIds = list(range(2, 10))
    
    tp = 0
    fn = 0
    fp = 0
    sum_iou = 0
    partial_box_num_total = 0
    missing_box_num_total = 0
    pseudo_box_num_total = 0
    N = len(image_ids)

    for i in tqdm(range(N)):
        im_idx = image_ids[i]
        imginfo = coco.loadImgs(im_idx)[0]
        image_width = imginfo['width']
        image_height = imginfo['height']

        # load annotations
        partial_anns = coco.loadAnns(coco.getAnnIds(imgIds=(im_idx,)))
        full_anns = coco_full.loadAnns(coco_full.getAnnIds(imgIds=(im_idx,), catIds=catIds))

        # obtain boxes
        pseudo_boxes = [obj["bbox"] for obj in partial_anns if "ispseudo" in obj.keys()]
        partial_boxes = [obj["bbox"] for obj in partial_anns if "ispseudo" not in obj.keys()]
        partial_boxes_id = set([obj["id"] for obj in partial_anns if "ispseudo" not in obj.keys()])
        missing_boxes = [obj["bbox"] for obj in full_anns if obj["id"] not in partial_boxes_id]

        partial_box_num = len(partial_boxes)
        missing_box_num = len(missing_boxes)
        pseudo_box_num = len(pseudo_boxes)

        partial_box_num_total += partial_box_num
        missing_box_num_total += missing_box_num
        pseudo_box_num_total += pseudo_box_num
        
        pseudo_boxes = convert_box_to_boxlist(pseudo_boxes, image_width, image_height)
        partial_boxes = convert_box_to_boxlist(partial_boxes, image_width, image_height)
        missing_boxes = convert_box_to_boxlist(missing_boxes, image_width, image_height)

        if missing_box_num == 0:
            fp += pseudo_box_num
        elif pseudo_box_num == 0:
            fn += missing_box_num
        else:
            # compute iou
            overlaps = boxlist_iou(missing_boxes, pseudo_boxes).numpy()
            matched_cnt = 0

            for i in range(missing_box_num):
                matched = np.argmax(overlaps[i])
                if overlaps[i, matched] >= 0.5:
                    tp += 1
                    sum_iou += overlaps[i, matched]
                    overlaps[:, matched] = 0
                    matched_cnt += 1
                else:
                    fn += 1
            
            fp += pseudo_box_num - matched_cnt

    print(tp, fp, fn, sum_iou/tp)
    print('PQ = ', sum_iou / (tp + 0.5*fp + 0.5*fn))
    print('partial_box_num_total:', partial_box_num_total)
    print('missing_box_num_total:', missing_box_num_total)
    print('pseudo_box_num_total:', pseudo_box_num_total)
