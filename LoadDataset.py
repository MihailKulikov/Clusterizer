from itertools import groupby
from pathlib import Path
from typing import List
import numpy as np
from catalyst.contrib.utils.cv.image import imread
from torchvision.transforms import ToTensor


def get_dataset(path: str) -> List[List[np.ndarray]]:
    all_images = list(Path(path).rglob("*.jpg"))
    grouped = groupby(all_images, lambda im_path: str(im_path.relative_to(path)).split('\\')[0])
    transform = ToTensor()
    data = [(list(map(lambda im_path: transform(imread(im_path))[None, :, :],
                      list(value)))) for _, value in grouped]
    return data
