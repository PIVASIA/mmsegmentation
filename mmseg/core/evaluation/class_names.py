# Copyright (c) OpenMMLab. All rights reserved.
import mmcv


def cityscapes_classes():
    """Cityscapes class names for external use."""
    return [
        'tenbien', 'tenduong'
    ]


def ade_classes():
    """ADE20K class names for external use."""
    return ['tenbien', 'tenduong']


def voc_classes():
    """Pascal VOC class names for external use."""
    return [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
        'tvmonitor'
    ]


def cocostuff_classes():
    """CocoStuff class names for external use."""
    return ['person', 'bicycle']


def loveda_classes():
    """LoveDA class names for external use."""
    return [
        'background', 'building', 'road', 'water', 'barren', 'forest',
        'agricultural'
    ]


def potsdam_classes():
    """Potsdam class names for external use."""
    return [
        'impervious_surface', 'building', 'low_vegetation', 'tree', 'car',
        'clutter'
    ]


def vaihingen_classes():
    """Vaihingen class names for external use."""
    return [
        'impervious_surface', 'building', 'low_vegetation', 'tree', 'car',
        'clutter'
    ]


def isaid_classes():
    """iSAID class names for external use."""
    return [
        'background', 'ship', 'store_tank', 'baseball_diamond', 'tennis_court',
        'basketball_court', 'Ground_Track_Field', 'Bridge', 'Large_Vehicle',
        'Small_Vehicle', 'Helicopter', 'Swimming_pool', 'Roundabout',
        'Soccer_ball_field', 'plane', 'Harbor'
    ]


def stare_classes():
    """stare class names for external use."""
    return ['background', 'vessel']


def cityscapes_palette():
    """Cityscapes palette for external use."""
    return [[38, 38, 38], [75, 75, 75]]


def ade_palette():
    """ADE20K palette for external use."""
    return [[38, 38, 38], [75, 75, 75]]


def voc_palette():
    """Pascal VOC palette for external use."""
    return [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
            [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
            [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
            [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]


def cocostuff_palette():
    """CocoStuff palette for external use."""
    return [[0, 192, 64], [0, 192, 64]]


def loveda_palette():
    """LoveDA palette for external use."""
    return [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
            [159, 129, 183], [0, 255, 0], [255, 195, 128]]


def potsdam_palette():
    """Potsdam palette for external use."""
    return [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
            [255, 255, 0], [255, 0, 0]]


def vaihingen_palette():
    """Vaihingen palette for external use."""
    return [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
            [255, 255, 0], [255, 0, 0]]


def isaid_palette():
    """iSAID palette for external use."""
    return [[0, 0, 0], [0, 0, 63], [0, 63, 63], [0, 63, 0], [0, 63, 127],
            [0, 63, 191], [0, 63, 255], [0, 127, 63], [0, 127,
                                                       127], [0, 0, 127],
            [0, 0, 191], [0, 0, 255], [0, 191, 127], [0, 127, 191],
            [0, 127, 255], [0, 100, 155]]


def stare_palette():
    """STARE palette for external use."""
    return [[120, 120, 120], [6, 230, 230]]


dataset_aliases = {
    'cityscapes': ['cityscapes'],
    'ade': ['ade', 'ade20k'],
    'voc': ['voc', 'pascal_voc', 'voc12', 'voc12aug'],
    'loveda': ['loveda'],
    'potsdam': ['potsdam'],
    'vaihingen': ['vaihingen'],
    'cocostuff': [
        'cocostuff', 'cocostuff10k', 'cocostuff164k', 'coco-stuff',
        'coco-stuff10k', 'coco-stuff164k', 'coco_stuff', 'coco_stuff10k',
        'coco_stuff164k'
    ],
    'isaid': ['isaid', 'iSAID'],
    'stare': ['stare', 'STARE']
}


def get_classes(dataset):
    """Get class names of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if mmcv.is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_classes()')
        else:
            raise ValueError(f'Unrecognized dataset: {dataset}')
    else:
        raise TypeError(f'dataset must a str, but got {type(dataset)}')
    return labels


def get_palette(dataset):
    """Get class palette (RGB) of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if mmcv.is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_palette()')
        else:
            raise ValueError(f'Unrecognized dataset: {dataset}')
    else:
        raise TypeError(f'dataset must a str, but got {type(dataset)}')
    return labels
