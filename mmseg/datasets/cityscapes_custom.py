from .builder import DATASETS
from .custom import CustomDataset
import os.path as osp

@DATASETS.register_module()
class CityscapesCustomDataset(CustomDataset):
  CLASSES = ('tenbien', 'tenduong')
  PALETTE = [[38, 38, 38], [75, 75, 75]]
  def __init__(self, split, **kwargs):
    super().__init__(img_suffix='.jpg', seg_map_suffix='.png', 
                     split=split, **kwargs)
    # assert osp.exists(self.img_dir) and self.split is not None