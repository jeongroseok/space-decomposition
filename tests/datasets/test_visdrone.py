from src.datasets.visdrone import *
from torchvision.datasets.folder import is_image_file
import os


def test_dataset():
    root = "D:\\datasets\\VisDrone2019-DET-train"
    dataset = VisDrone(root)
    assert len(dataset) == 6471


def test_parse_annotation():
    root = "D:\\datasets\\VisDrone2019-DET-train"
    annotation_dir = os.path.join(root, "annotations")
    filenames = os.listdir(annotation_dir)
    filename = os.path.join(annotation_dir, filenames[0])
    ret = parse_annotation(filename)
    assert len(ret) > 0