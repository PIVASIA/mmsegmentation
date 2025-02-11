# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class COCOStuffDataset(CustomDataset):
    """COCO-Stuff dataset.

    In segmentation map annotation for COCO-Stuff, Train-IDs of the 10k version
    are from 1 to 171, where 0 is the ignore index, and Train-ID of COCO Stuff
    164k is from 0 to 170, where 255 is the ignore index. So, they are all 171
    semantic categories. ``reduce_zero_label`` is set to True and False for the
    10k and 164k versions, respectively. The ``img_suffix`` is fixed to '.jpg',
    and ``seg_map_suffix`` is fixed to '.png'.
    """
    CLASSES = (
        'person', 'bicycle')

    PALETTE = [[0, 192, 64], [0, 192, 64]]

    def __init__(self, **kwargs):
        super(COCOStuffDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='_labelTrainIds.png', **kwargs)
