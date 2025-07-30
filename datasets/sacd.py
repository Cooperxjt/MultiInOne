
import json
import os
from pathlib import Path

import cv2
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, DistributedSampler

from .sacd_transforms import CropTransform


def combine_tensors(input_data):
    output_data = {}

    for key, value in input_data.items():
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], list):
            # 将多个 tensor 合并为一个 tensor
            tensor_list = [torch.tensor(box) for box in value]
            output_data[key] = torch.stack(tensor_list)
        else:
            # 其他类型的数据保持不变
            output_data[key] = value

    return output_data


class SACDDataset(data.Dataset):
    def __init__(self, root, image_set, img_list, ann_list, transforms):

        self.root = root
        self.set = image_set
        self.img_list = img_list
        self.ann_list = ann_list
        self.transforms = transforms

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.img_list[idx])
        anno_path = os.path.join(self.root, self.ann_list[idx])

        # print(img_path)
        # print(anno_path)

        image = cv2.imread(img_path)

        orig_size = torch.as_tensor(list(image.shape[:2]))

        target = {}

        with open(anno_path, 'r') as f:
            content = f.read()

            data = json.loads(content)

            f.close()

            target['phase1_box'] = data['phase1_box']  # x1 y1 x2 y2
            target['phase2_box'] = data['phase2_box']

            # target['select_from_phase2_box'] = data['select_from_phase2_box']
            # target['crop_box'] = data['crop_box']
            # target['best_box'] = data['best_box']

            target['crop_box'] = data['crop_box'] + data['best_box']

            target['mask_box'] = [json.loads(data['mask_box'])]  # x y w h
            target['mask_box'][0][2] = target['mask_box'][0][0] + target['mask_box'][0][2]
            target['mask_box'][0][3] = target['mask_box'][0][1] + target['mask_box'][0][3]

        target['image_id'] = torch.as_tensor(
            int(img_path.split('/')[-1].split('.')[0])
        )

        combine_target = combine_tensors(target)

        if self.transforms is not None:
            image, target = self.transforms(image, combine_target)

        target['labels'] = torch.zeros(target['crop_box'].shape[0]).type(torch.LongTensor)
        target['orig_size'] = orig_size

        return image, target


def build(image_set, args):
    root = Path(args.sacd_path)
    assert root.exists(), f'provided SACD path {root} does not exist'

    img_json = os.path.join(root, image_set+'.json')

    f = open(img_json, 'r')
    content = f.read()
    json_data = json.loads(content)
    f.close()

    img_list = json_data['imgs']
    ann_list = json_data['jsons']

    dataset = SACDDataset(
        root,
        image_set,
        img_list,
        ann_list,
        transforms=CropTransform()
    )

    return dataset
