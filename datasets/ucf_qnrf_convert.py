from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from scipy.io import loadmat

from PIL import Image, ImageDraw


class DrawExt(ImageDraw.ImageDraw):
    def __init__(self, draw: ImageDraw.ImageDraw) -> None:
        self.draw = draw

    def dot(self, xy: Tuple[int, int], fill: Tuple[int, int, int],
            radius: int) -> None:
        x, y = xy
        x0, x1 = x - radius, x + radius
        y0, y1 = y - radius, y + radius
        self.draw.ellipse((x0, y0, x1, y1), fill)


def main():
    im = Image.open("data\\UCF-QNRF_ECCV18\\Train\\img_0001.jpg")
    draw = DrawExt(ImageDraw.Draw(im))

    mat_dict = loadmat("data\\UCF-QNRF_ECCV18\\Train\\img_0001_ann.mat")
    for point in mat_dict['annPoints']:
        draw.dot(point, fill=(255, 0, 0), radius=25)

    plt.imshow(im)
    plt.show()


if __name__ == "__main__":
    main()