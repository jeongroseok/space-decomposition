import os
from typing import Any, Callable, Dict, List, Optional, Tuple
from torch.utils.data import Dataset
from PIL import Image
from torchvision.datasets.folder import is_image_file


def list_image(path: str) -> List[str]:
    return list(filter(is_image_file, os.listdir(path)))


def parse_annotation(filename: str, size: Tuple[int, int]) -> List[Tuple]:
    lines = open(filename).read().splitlines()
    ret = []
    for line in lines:
        v = line.split(',')
        left = int(v[0]) / size[0]
        top = int(v[1]) / size[1]
        width = int(v[2]) / size[0]
        height = int(v[3]) / size[1]
        label = int(v[5])
        ret += [(label, left, top, left + width, top + height)]
    return ret


class VisDrone(Dataset):
    classes = {
        0: "ignored_regions",
        1: "pedestrian",
        2: "people",
        3: "bicycle",
        4: "car",
        5: "van",
        6: "truck",
        7: "tricycle",
        8: "awning_tricycle",
        9: "bus",
        10: "motor",
        11: "others"
    }

    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        self.__root = root
        self.__transform = transform
        self.__target_transform = target_transform

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        self.__filenames = list_image(os.path.join(self.__root, "images"))

    def __len__(self) -> int:
        return len(self.__filenames)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Image.Image, List[Tuple[float, float, float, float]]]:
        item = self.__filenames[idx]
        img_path = os.path.join(self.__root, "images", f"{item}")
        anno_path = os.path.join(self.__root, "annotations",
                                 f"{os.path.splitext(item)[0]}.txt")

        with open(img_path, 'rb') as f:
            sample = Image.open(f)
            sample.convert('RGB')
            target = parse_annotation(anno_path, sample.size)

            if self.__transform is not None:
                sample = self.__transform(sample)
            if self.__target_transform is not None:
                target = self.__target_transform(target)

            return sample, target

    def _check_exists(self) -> bool:
        files = ["annotations", "images"]
        return all(
            [os.path.exists(os.path.join(self.__root, f)) for f in files])
