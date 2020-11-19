import os
from typing import Any, Callable, Dict, List, Optional, Tuple
from torch.utils.data import Dataset
from PIL import Image


def parse_annotation(filename: str, size: Tuple[int, int]) -> List[Tuple]:
    lines = open(filename).read().splitlines()[1:]
    ret = []
    for line in lines:
        v = line.split(' ')
        label = int(v[0])
        x0 = int(v[1]) / size[0]
        y0 = int(v[2]) / size[1]
        x1 = int(v[3]) / size[0]
        y1 = int(v[4]) / size[1]
        ret += [(label, x0, y0, x1, y1)]
    return ret


class WiderPerson(Dataset):
    classes = {
        1: "pedestrians",
        2: "riders",
        3: "partially-visible persons",
        4: "ignore regions",
        5: "crowd",
    }

    def __init__(self,
                 root: str,
                 split: str = "train",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        self.__root = root
        self.__split = split
        self.__transform = transform
        self.__target_transform = target_transform

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        subset_list_file = os.path.join(self.__root, f"{self.__split}.txt")
        self.__subset_list = open(subset_list_file).read().splitlines()

    def __len__(self) -> int:
        return len(self.__subset_list)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Image.Image, List[Tuple[float, float, float, float]]]:
        item = self.__subset_list[idx]
        img_path = os.path.join(self.__root, "Images", f"{item}.jpg")
        anno_path = os.path.join(self.__root, "Annotations", f"{item}.jpg.txt")

        with open(img_path, 'rb') as f:
            sample = Image.open(f)
            sample.convert('RGB')
            sample.convert('RGB')
            target = parse_annotation(anno_path, sample.size)

            if self.__transform is not None:
                sample = self.__transform(sample)
            if self.__target_transform is not None:
                target = self.__target_transform(target)

            return sample, target

    def _check_exists(self) -> bool:
        # two subset list, annotations folder, images folder
        files = ["train.txt", "val.txt", "Annotations", "Images"]
        return all(
            [os.path.exists(os.path.join(self.__root, f)) for f in files])
