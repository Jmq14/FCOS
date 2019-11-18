# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
from tqdm import tqdm
import numpy as np

from fcos_core.config import cfg
from fcos_core.modeling.poolers import Pooler
from fcos_core.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str


def compute_on_dataset(model, data_loader, device, pooler, timer=None, output_folder=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")

    if output_folder:
        gt_feature_output_dir = os.path.join(output_folder, "ground_truth_feature")
        pred_feature_output_dir = os.path.join(output_folder, "prediction_feature")
        os.makedirs(gt_feature_output_dir, exist_ok=True)
        os.makedirs(pred_feature_output_dir, exist_ok=True)
    
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids, boxes_ids = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            features, predictions = model(images.to(device))
            if timer:
                torch.cuda.synchronize()
                timer.toc()
            p3_features = features[0]
            # predictions = [prediction.to(cpu_device) for prediction in predictions]
            targets = [target.to(device) for target in targets]
            gt_box_features = pooler([p3_features], targets).to(cpu_device)
            pred_box_features = pooler([p3_features], predictions).to(cpu_device)
        
        flatten_boxes_id = [item for sublist in boxes_ids for item in sublist]
        if output_folder:
            for box_id, box in zip(flatten_boxes_id, gt_box_features):
                path = os.path.join(gt_feature_output_dir, "{}.npz".format(box_id))
                np.savez_compressed(path, feature=box)

            cnt = 0
            for i, image_id in enumerate(image_ids):
                path = os.path.join(pred_feature_output_dir, "{}.npz".format(image_id))
                box_num = len(predictions[i])
                np.savez_compressed(path, feature=pred_box_features[cnt:cnt+box_num,:,:,:])
                cnt += box_num

        results_dict.update(
            {img_id: (result, target_box_ids) for img_id, result, target_box_ids in zip(image_ids, predictions, boxes_ids)}
        )
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))

    # convert to a list
    box_ids = [box_id for i in image_ids for box_id in predictions[i][1]]
    predictions = [predictions[i][0] for i in image_ids]
    # print(predictions)
    return predictions, image_ids, box_ids


def get_box_feature(
        model,
        data_loader,
        dataset_name,
        device="cuda",
        output_folder=None,
        resolution=14,
        scales=(1./8.,),
        sampling_ratio=0,
        expected_results=(),
        expected_results_sigma_tol=4,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("fcos_core.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()

    # ROI align
    pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

    predictions = compute_on_dataset(model, data_loader, device, pooler, inference_timer, output_folder)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions, image_ids, box_ids = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))
        torch.save(box_ids, os.path.join(output_folder, 'box_ids.pth'))
        torch.save(image_ids, os.path.join(output_folder, 'image_ids.pth'))

    extra_args = dict(
        box_only=False,
        iou_types=("bbox",),
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
