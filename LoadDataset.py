from itertools import groupby
from pathlib import Path
from typing import Dict
from typing import List


def get_dataset(path: str) -> Dict[int, List[Path]]:
    all_images = list(Path(path).rglob("*.jpg"))
    grouped = groupby(all_images, lambda im_path: str(im_path.relative_to(path)).split('\\')[0])
    return {int(key): list(value) for (key, value) in grouped}

