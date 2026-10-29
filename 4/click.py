import json
import os.path

import matplotlib.pyplot as plt
import numpy as np

os.makedirs("input images/points", exist_ok=True)


def read_img(path: str) -> np.ndarray:
    image = plt.imread(path)
    if image.dtype == np.uint8:
        image = image / 255
    return image[:, :, :3]


def click_points(img_path: str, n: int) -> np.ndarray:
    img = read_img(img_path)
    plt.imshow(img)
    plt.title(f"Click {n} points")
    pts = plt.ginput(n, timeout=0)
    plt.close()
    return np.array(pts)


def save_points(path: str, pts1: np.ndarray, pts2: np.ndarray):
    with open(path, "w") as f:
        json.dump({"im1": pts1.tolist(), "im2": pts2.tolist()}, f)


def click_pair(path1: str, path2: str, out: str, n=10):
    if os.path.exists(out):
        return
    pts1 = click_points(path1, n)
    pts2 = click_points(path2, n)
    save_points(out, pts1, pts2)


def click_single(path: str, out: str, n: int = 4):
    if os.path.exists(out):
        return
    pts = click_points(path, n)
    with open(out, "w") as f:
        json.dump({"im1": pts.tolist()}, f)


if __name__ == "__main__":
    click_single("input images/resized/rectifications/rectification_1.jpg",
                 "input images/points/rectification_1.json")

    click_single("input images/resized/rectifications/rectification_2.jpg",
                 "input images/points/rectification_2.json")

    click_pair("input images/resized/mosaics/mosaic_1_a.jpg",
               "input images/resized/mosaics/mosaic_1_b.jpg",
               "input images/points/mosaic_1_a_b.json")

    click_pair("input images/resized/mosaics/mosaic_1_b.jpg",
               "input images/resized/mosaics/mosaic_1_c.jpg",
               "input images/points/mosaic_1_b_c.json")

    click_pair("input images/resized/mosaics/mosaic_2_a.jpg",
               "input images/resized/mosaics/mosaic_2_b.jpg",
               "input images/points/mosaic_2_a_b.json")

    click_pair("input images/resized/mosaics/mosaic_2_b.jpg",
               "input images/resized/mosaics/mosaic_2_c.jpg",
               "input images/points/mosaic_2_b_c.json")

    click_pair("input images/resized/mosaics/mosaic_3_a.jpg",
               "input images/resized/mosaics/mosaic_3_b.jpg",
               "input images/points/mosaic_3_a_b.json")

    click_pair("input images/resized/mosaics/mosaic_3_b.jpg",
               "input images/resized/mosaics/mosaic_3_c.jpg",
               "input images/points/mosaic_3_b_c.json")
