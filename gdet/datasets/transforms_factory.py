import torch
from packaging import version
from typing import Sequence

from gdet.datasets import transforms
from gdet.datasets.transforms import construct_transforms
from gdet.registries import PIPELINES


def data_transforms_cls(cfg):
    cfg_data = cfg.dataset
    data_aug = cfg_data.data_augmentation
    aug_args = cfg.data_augmentation_args

    operations = {
        'random_crop': random_apply(
            transforms.RandomResizedCrop(
                size=(cfg_data.input_size, cfg_data.input_size),
                scale=aug_args.random_crop.scale,
                ratio=aug_args.random_crop.ratio
            ),
            p=aug_args.random_crop.prob
        ),
        'horizontal_flip': transforms.RandomHorizontalFlip(
            p=aug_args.horizontal_flip.prob
        ),
        'vertical_flip': transforms.RandomVerticalFlip(
            p=aug_args.vertical_flip.prob
        ),
        'color_distortion': random_apply(
            transforms.ColorJitter(
                brightness=aug_args.color_distortion.brightness,
                contrast=aug_args.color_distortion.contrast,
                saturation=aug_args.color_distortion.saturation,
                hue=aug_args.color_distortion.hue
            ),
            p=aug_args.color_distortion.prob
        ),
        'rotation': random_apply(
            transforms.RandomRotation(
                degrees=aug_args.rotation.degrees,
                fill=aug_args.value_fill
            ),
            p=aug_args.rotation.prob
        ),
        'translation': random_apply(
            transforms.RandomAffine(
                degrees=0,
                translate=aug_args.translation.range,
                fill=aug_args.value_fill
            ),
            p=aug_args.translation.prob
        ),
        'grayscale': transforms.RandomGrayscale(
            p=aug_args.grayscale.prob
        )
    }

    if version.parse(torch.__version__) >= version.parse('1.7.1'):
        operations['gaussian_blur'] = random_apply(
            transforms.GaussianBlur(
                kernel_size=aug_args.gaussian_blur.kernel_size,
                sigma=aug_args.gaussian_blur.sigma
            ),
            p=aug_args.gaussian_blur.prob
        )

    augmentations = []
    for op in data_aug:
        if op not in operations:
            raise NotImplementedError('Not implemented data augmentation operations: {}'.format(op))
        augmentations.append(operations[op])

    normalization = [
        transforms.TF_Resize((cfg_data.input_size, cfg_data.input_size)),
        transforms.ToTensor(),
        transforms.TF_Normalize(cfg_data.mean, cfg_data.std)
    ]

    train_preprocess = transforms.Compose([
        *augmentations,
        *normalization
    ])

    test_preprocess = transforms.Compose(normalization)

    return train_preprocess, test_preprocess


def random_apply(op, p):
    return transforms.RandomApply([op], p=p)


def simple_transform(input_size):
    return transforms.Compose([
        transforms.TF_Resize((input_size, input_size)),
        transforms.ToTensor()
    ])
