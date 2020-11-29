from itertools import groupby
from pathlib import Path
from typing import List
import numpy as np
from catalyst.contrib.utils.cv.image import imread


def get_dataset(path: str) -> List[List[np.ndarray]]:
    all_images = list(Path(path).rglob("*.jpg"))
    grouped = groupby(all_images, lambda im_path: str(im_path.relative_to(path)).split('\\')[0])
    data = [(list(map(lambda im_path: imread(im_path), list(value)))) for _, value in grouped]
    return np.array(data, dtype=object)
