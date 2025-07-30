"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
from PIL import Image, ImageDraw

import torch

import util.misc as utils
from datasets.sacd_eval import sacd_evaluator


def train_one_epoch(context_model: torch.nn.Module, model: torch.nn.Module,
                    criterion: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, num_queries: int = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: (v.to(device)if k != "img_path" else v) for k, v in t.items()} for t in targets]

        # 从 所有的 300 个 query 中随机选取一部分
        context_query, context_outputs = context_model(samples)
        batch_size, total_queries, _ = context_query.shape
        random_indices = torch.randperm(total_queries)[:num_queries]
        random_pred_boxes = context_outputs['pred_boxes'][:, random_indices, :]
        random_context_queries = context_query[:, random_indices, :]

        _, outputs = model(samples, random_context_queries)

        # _, outputs = model(samples)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f'{k}_unscaled': v
            for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(context_model, model, criterion, postprocessors, data_loader, num_queries, set, device, output_dir, epoch=0, save_crop=False):

    model.eval()
    criterion.eval()
    context_model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    evaluator = sacd_evaluator(
        output_dir, device=device, set=set, num_queries=num_queries
    )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: (v.to(device)if k != "img_path" else v) for k, v in t.items()} for t in targets]

        context_query, context_outputs = context_model(samples)
        batch_size, total_queries, _ = context_query.shape
        random_indices = torch.randperm(total_queries)[:num_queries]
        random_pred_boxes = context_outputs['pred_boxes'][:, random_indices, :]
        random_context_queries = context_query[:, random_indices, :]

        _, outputs = model(samples, random_context_queries)
        # _, outputs = model(samples)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}

        loss_dict_reduced_unscaled = {
            f'{k}_unscaled': v
            for k, v in loss_dict_reduced.items()
        }

        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results, results_oriorder = postprocessors['bbox'](outputs, orig_target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if save_crop:
            for key, value in res.items():
                for idx in range(10):
                    img_path = targets[0]['img_path']
                    pred_boxes = res[key]['boxes'][idx]
                    pred_reference = res[key]['reference_points'][idx]

                    output_dir = 'output_txt_files'
                    os.makedirs(output_dir, exist_ok=True)

                    txt_file_path = os.path.join(output_dir, f"{os.path.basename(img_path).split('.')[0]}.txt")

                    with open(txt_file_path, 'w') as f:
                        # Write img_path to the file
                        f.write(f"Image Path: {img_path}\n")

                        # Write pred_boxes to the file
                        f.write("Predicted Boxes:\n")
                        for box in res[key]['boxes']:
                            f.write(f"{box}\n")

                    image = Image.open(img_path)
                    draw = ImageDraw.Draw(image)

                    draw.rectangle(
                        [(int(pred_boxes[0]), int(pred_boxes[1])), (int(pred_boxes[2]), int(pred_boxes[3]))],
                        outline="red",
                        width=3
                    )

                    ref_x, ref_y = int(pred_reference[0]), int(pred_reference[1])
                    draw.ellipse(
                        [(ref_x - 5, ref_y - 5), (ref_x + 5, ref_y + 5)],
                        fill="blue",
                        outline="blue"
                    )

                    crop_folder = img_path.split('/')[-1].split('.')[0]

                    if not os.path.exists(crop_folder):
                        os.mkdir(crop_folder)

                    output_path = os.path.join(crop_folder, str(idx) + '_' + img_path.split('/')[-1])
                    image.save(output_path)

        if evaluator is not None:
            evaluator.update(res, targets)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # accumulate predictions from all images
    if evaluator is not None:
        evaluator.iou_cal()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats, evaluator
