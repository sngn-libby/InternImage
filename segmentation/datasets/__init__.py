from os import path
import numpy as np

from torchvision.datasets.cifar import CIFAR100
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


def build_palette(num=256):
    def bit_get(val, idx):
        return (val >> idx) & 1

    colormap = np.zeros((num, 3), dtype=int)
    ind = np.arange(num, dtype=int)

    for shift in reversed(list(range(8))):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    return colormap


@DATASETS.register_module()
class Cifar100Dataset(CustomDataset):
    with open("D:/datasets/cifar/cifar100/classes.txt", "r") as f:
        lines = f.readlines()
    CLASSES = [line.strip() for line in lines]
    # print(f":: Log :: CLASSES - {CLASSES}")
    PALETTE = build_palette(len(CLASSES))

    def __init__(self, split=None, **kwargs):
        print(f":: Log :: Cifar100Dataset Registered in mmseg")
        super().__init__(img_suffix='.png', seg_map_suffix='.png',
                         split=split, **kwargs)
        assert path.exists(self.img_dir)


