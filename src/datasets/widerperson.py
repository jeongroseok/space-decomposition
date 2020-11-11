import os
from typing import Any, Callable, Dict, Optional, Tuple
from torch.utils.data import Dataset
from PIL import Image


class WiderPerson(Dataset):
    def __init__(self,
                 root: str,
                 split: str = "train",
                 transform: Optional[Callable] = None):
        self.__root = root
        self.__transform = transform
        self.__split = split

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        self.classes = {
            1: "pedestrians",
            2: "riders",
            3: "partially-visible persons",
            4: "ignore regions",
            5: "crowd",
        }
        subset_list_file = os.path.join(self.__root, f"{self.__split}.txt")
        self.__subset_list = open(subset_list_file).read().splitlines()

    def __len__(self) -> int:
        return len(self.__subset_list)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        item = self.__subset_list[idx]
        img_path = os.path.join(self.__root, "Images", f"{item}.jpg")
        anno_path = os.path.join(self.__root, "Annotations", f"{item}.jpg.txt")

        target = self._read_anno(anno_path)

        with open(img_path, 'rb') as f:
            sample = Image.open(f)
            sample.convert('RGB')
            size = sample.size
            if self.__transform is not None:
                sample = self.__transform(sample)
            return sample, [(t[0], t[1] / size[0], t[2] / size[1],
                             t[3] / size[0], t[4] / size[1]) for t in target]

    def _read_anno(self, path: str) -> Dict:
        lines = open(path).read().splitlines()[1:]
        return [[int(v) for v in line.split(' ')] for line in lines]

    def _check_exists(self) -> bool:
        files = ["train.txt", "val.txt", "Annotations", "Images"]
        return all(
            [os.path.exists(os.path.join(self.__root, f)) for f in files])
