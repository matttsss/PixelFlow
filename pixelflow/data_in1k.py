# ImageNet-1K Dataset and DataLoader

import torch
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
from functools import partial
import numpy as np

# https://github.com/facebookresearch/DiT/blob/main/train.py#L85
def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def collate_fn(examples, pipeline):
    pixel_values = torch.stack([eg[0] for eg in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = [eg[1] for eg in examples]
    return pipeline.prepare_training_batch(pixel_values=pixel_values, input_ids=input_ids)


def build_imagenet_loader(config, pipeline):
    if config.data.center_crop:
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, config.data.resolution)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(round(config.data.resolution * config.data.expand_ratio), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.RandomCrop(config.data.resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    dataset = ImageFolder(config.data.root, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
        shuffle=True,
        seed=config.seed,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        collate_fn=partial(collate_fn, pipeline=pipeline),
        shuffle=False,
        sampler=sampler,
        num_workers=config.data.num_workers,
        drop_last=True,
    )
    return loader, sampler
