import numpy as np
from tqdm import tqdm
import json
import os

import torch
import faiss

from pycocotools.coco import COCO


def build_index(feat, d):
    cpu_index = faiss.IndexFlatL2(d)
    cpu_index.add(feat)
    return cpu_index


def find_nearest_neighbor(index, query_feat):
    D, I = index.search(query_feat, 1)
    D /= query_feat.shape[1]
    return D.reshape(-1).tolist(), I.reshape(-1).tolist()


if __name__ == "__main__":
    feature_dir = '/home/mengqinj/capstone/output/stage1/coco_2017_train_partial/'
    anno_ids = torch.load(feature_dir + 'box_ids.pth')
    image_ids = torch.load(feature_dir + 'image_ids.pth')
    predictions = torch.load(feature_dir + 'predictions.pth')

    dataDir=''
    annFile='datasets/coco/annotations/instances_train2017_0.5.json'
    coco=COCO(annFile)
    raw_image_ids=sorted(coco.getImgIds())

    feat = []
    print('Loading ground truth features...')
    for i in tqdm(anno_ids):
        try:
            roi = np.load(os.path.join(feature_dir, 'ground_truth_feature/{}.npz'.format(i)))['feature']
        except:
            print(i)
        feat.append(roi.reshape(-1))
    feat = np.stack(feat, axis=0)
    print(feat.shape)

    d = 256 * 7 * 7
    print('Building database index...')
    index = build_index(feat, d)

    results = []
    print('Loading prediction features and querying...')
    i = 0
    for i in tqdm(image_ids):
        roi = np.load(os.path.join(feature_dir, 'prediction_feature/{}.npz'.format(i)))['feature']
        
        boxlist = predictions[i]

        img_info = coco.loadImgs(raw_image_ids[i])[0]
        image_width = img_info["width"]
        image_height = img_info["height"]
        boxlist = boxlist.resize((image_width, image_height))
        boxlist = boxlist.convert("xywh")

        bboxes = boxlist.bbox.tolist()
        labels = boxlist.get_field('labels').tolist()
        scores = boxlist.get_field('scores').tolist()
        if len(roi.shape) > 0 and roi.shape[0] > 0:
            if len(roi.shape) >= 4:
                query_feat = roi.reshape(roi.shape[0], -1)
            else:
                query_feat = roi.reshape(1, -1)
            # print(query_feat.shape[0], len(boxlist))
            D, I = find_nearest_neighbor(index, query_feat)
            
            for j in range(len(boxlist)):
                results.append({
                    'image_id': i,
                    'NN_distance': D[j],
                    'NN_id': I[j],
                    'score': scores[j],
                    'bbox': bboxes[j],
                    'category_id': labels[j],
                })

    # print(results)
    with open('results.json', 'w') as f:
        json.dump(results, f)