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


def generate_pseudo_label_with_confidence_score(boxes, image_id, score_thre):
    scores = boxes.get_field("scores")
    _, idx = scores.sort(0, descending=True)
    if isinstance(score_thre, float):
        keep = torch.nonzero(scores >= score_thre).squeeze(1)
    else:
        labels = boxes.get_field("labels")
        keep = torch.nonzero(scores >= score_thre[labels]).squeeze(1)
        
    return idx[:len(keep)]

def parse_predictions():
    pass

def new_annotation_json(pseudo_labels, img_id, ann_id):
    labels = pseudo_labels.get_field("labels").tolist()
    boxes = pseudo_labels.convert("xywh").bbox
    annos = []
    for box, c in zip(boxes, labels):
        annos.append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": c,
            "bbox": box.tolist(),
            "segmentation": [[0., 0.]],
            "area": float(box[2] * box[3]),
            "iscrowd": 0,
            "ispseudo": True,
        })
        ann_id = ann_id + 1
    return annos, ann_id


def main(args):
    annFile = 'datasets/coco/annotations/instances_train2017_0.5.json'
    coco = COCO(annFile)

    with open(annFile, 'r') as f:
        result_json = json.load(f)
    annos_json = result_json['annotations']
    # anno_id = max([ann['id'] for ann in annos_json]) + 1

    output_dir = os.path.join(args.predictions, 'coco_2017_train_partial')
    image_ids = torch.load(os.path.join(output_dir, 'image_ids.pth'))
    predictions = torch.load(os.path.join(output_dir, 'predictions.pth'))
    anno_id = max(torch.load(os.path.join(output_dir, 'box_ids.pth'))) + 1

    imgIds=sorted(coco.getImgIds())
    
    threshold = args.confidence
    # threshold = torch.tensor([-1.0, 0.46633365750312805, 0.4409848749637604, 0.47267603874206543, 0.4707889258861542, 0.5220812559127808, 0.5358721613883972, 0.5226702690124512, 0.45160290598869324])
    iou_threshold = 0.5

    cpu_device = torch.device("cpu")

    partial_box_num = 0

    N = len(image_ids)
    for i in tqdm(range(N)):
        im_idx = image_ids[i]
        bbox = predictions[i]
        imginfo = coco.loadImgs(imgIds[im_idx])[0]
        image_width = imginfo['width']
        image_height = imginfo['height']

        # load annotations
        partial_anns = coco.loadAnns(coco.getAnnIds(imgIds=(imgIds[im_idx],)))
        # full_anns = coco_full.loadAnns(coco_full.getAnnIds(imgIds=(imgIds[im_idx],), catIds=catIds))

        partial_boxes = [obj["bbox"] for obj in partial_anns]
        partial_boxes_ids = set([obj["id"] for obj in partial_anns])

        partial_boxes = torch.as_tensor(partial_boxes).reshape(-1, 4)  # guard against no boxes
        partial_boxes = BoxList(partial_boxes, (image_width, image_height), mode="xywh").convert(
            "xyxy"
        )

        partial_box_num += len(partial_boxes_ids)

        # get predictions
        bbox = bbox.resize((image_width, image_height))
        bbox = bbox.to(cpu_device)

        # generate pseudo labels
        idx = generate_pseudo_label_with_confidence_score(bbox, im_idx, threshold)

        if len(idx) > 0:
            pseudo_labels = bbox[idx]
            scores = pseudo_labels.get_field("scores").tolist()

            # compute iou
            overlaps = boxlist_iou(partial_boxes, pseudo_labels)
            matched_id = [True] * len(pseudo_labels)

            # remove predictions for partial labels
            for i in range(len(partial_boxes)):
                matched = np.argmax(overlaps[i])
                if overlaps[i, matched] >= iou_threshold:
                    matched_id[matched] = False

            pseudo_labels = pseudo_labels[matched_id]
            # print(num, len(pseudo_labels))
            pseudo_annos, anno_id = new_annotation_json(pseudo_labels, imgIds[im_idx], anno_id)
            annos_json.extend(pseudo_annos)

    print('confidence threshold: {}'.format(threshold))

    result_json['annotations'] = annos_json
    with open(args.annotation, 'w') as f:
        json.dump(result_json, f)

    print(partial_box_num, len(result_json['annotations']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", help="prediction directory path. e.g output/stage1/",
                    type=str, default="/home/mengqinj/capstone/output/stage1/")
    parser.add_argument("--annotation", help="output annotation path. e.g instances_train_2017.json",
                    type=str, default="instances_train_2017.json")
    parser.add_argument("--confidence", help="confidence score threshold",
                    type=float, default=0.5)
    args = parser.parse_args()
    main(args)