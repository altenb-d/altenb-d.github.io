import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import cv2 as cv


def read_img(file_path: str) -> np.ndarray:
    """
    Reads an image from the given file path without any conversion.

    The image is loaded unchanged, so single-channel plates stay two-dimensional
    and 16-bit TIFF files keep their full bit depth.
    Exits the program if the file cannot be read.

    :param file_path: The file path of the image.
    :return: The image as 2D numpy array, dtype uint8 or uint16.
    """
    img = cv.imread(file_path, cv.IMREAD_UNCHANGED)

    if img is None:
        sys.exit("Can't find image under given path!")
    return img


def to_float(img: np.ndarray) -> np.ndarray:
    """
    Converts an image to float64 with values scaled to the range [0, 1].

    The scaling factor is derived from the input dtype, not from the image content,
    so that all images share the same value range regardless of bit depth.

    :param img: The image as numpy array, dtype uint8 or uint16.
    :return: The image as float64 array with values in [0, 1].
    :raises ValueError: If the dtype is neither uint8 nor uint16.
    """
    data_type = img.dtype
    img = img.astype(np.float64)
    if data_type == np.uint8:
        img = img / 255
    elif data_type == np.uint16:
        img = img / 65535
    else:
        raise ValueError(f"Incompatible data type: {data_type}")
    return img


def split_channels(plate: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits a glass plate into its three color channels.

    The channels are stacked vertically on the plate in the order blue, green, red
    from top to bottom. If the plate height is not divisible by three, the remaining
    rows at the bottom are discarded so that all three channels have identical shape.

    :param plate: The full glass plate as 2D numpy array.
    :return: A tuple (blue, green, red) of three equally sized 2D arrays.
    """
    image_height = plate.shape[0]
    channel_height = image_height // 3
    blue = plate[:channel_height, :]
    green = plate[channel_height: 2 * channel_height, :]
    red = plate[2 * channel_height: 3 * channel_height, :]
    return blue, green, red


def stack_channels(channels: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Stacks three color channels into a single color image.

    The channels are stacked in the order given, so passing the output of
    split_channels directly produces a BGR image as expected by OpenCV.

    :param channels: A tuple of three equally sized 2D arrays.
    :return: A 3D array of shape (height, width, 3).
    """
    stacked = np.dstack(channels)
    return stacked


def crop_fraction(img: np.ndarray, fraction: float = 0.1) -> np.ndarray:
    """
    Removes a fraction from all four sides of an image.

    Used before scoring so that the plate borders, which are identical across
    all three channels, do not dominate the matching metric.

    :param img: The image as numpy array.
    :param fraction: The fraction to remove per side, e.g. 0.1 for 10%.
    :return: The cropped image.
    """
    height = img.shape[0]
    width = img.shape[1]

    margin_y = int(height * fraction)
    margin_x = int(width * fraction)
    return img[margin_y: height - margin_y, margin_x: width - margin_x]


def ncc(a: np.ndarray, b: np.ndarray) -> np.float64:
    """
    Computes the Normalized Cross-Correlation between two images.

    Both images are mean-subtracted and normalized to unit length before the dot
    product is taken. A higher value means a better match, with 1.0 for identical
    images.

    :param a: The first image as numpy array.
    :param b: The second image as numpy array, same shape as a.
    :return: The NCC score, in the range [-1, 1].
    """
    a_centered = a - np.mean(a)
    a_norm = a_centered / np.linalg.norm(a_centered)
    b_centered = b - np.mean(b)
    b_norm = b_centered / np.linalg.norm(b_centered)

    return np.sum(a_norm * b_norm)


def l2(a: np.ndarray, b: np.ndarray) -> np.float64:
    """
    Computes the negated Euclidean distance between two images.

    The result is negated so that, as with ncc, a higher value means a better match.
    This allows both metrics to be used interchangeably without a case
    distinction at the call site.

    :param a: The first image as numpy array.
    :param b: The second image as numpy array, same shape as a.
    :return: The negative L2 distance, 0.0 for identical images.
    """
    return -np.sqrt(np.sum((a - b) ** 2))


def align_single(reference_channel: np.ndarray,
                 moving_channel: np.ndarray,
                 radius: int,
                 metric,
                 crop: float = 0.1) -> tuple[int, int]:
    """
    Finds the best translation of one channel relative to another.

    Exhaustively searches all displacements within [-radius, radius] in both directions,
    scores each one with the given metric and returns the best. Scoring is done on the
    cropped interior of both images so that the wrapped-around edge introduced by the shift
    is ignored.

    :param reference_channel: The channel to align against, as 2D array.
    :param moving_channel: The channel to be shifted, as 2D array.
    :param radius: The maximum displacement searched in each direction.
    :param metric: A function taking two arrays and returning a score, where a
        higher value means a better match.
    :param crop: The fraction to remove per side, e.g. 0.1 for 10%, default 0.1.
    :return: The best displacement as tuple (dy, dx).
    """
    reference_channel_cropped = crop_fraction(reference_channel, crop)

    best_score = -np.inf
    best_offset = (0, 0)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            rolled = np.roll(moving_channel, (dy, dx), (0, 1))
            moving_channel_cropped = crop_fraction(rolled, crop)
            score = metric(reference_channel_cropped, moving_channel_cropped)
            if score > best_score:
                best_score = score
                best_offset = (dy, dx)

    return best_offset


def downsample(image: np.ndarray) -> np.ndarray:
    """
    Reduces an image to half its width and height.

    A Gaussian blur is applied before discarding every second row and column.
    Without this low-pass step, high frequencies in the original would alias into
    the smaller image and the coarse pyramid levels would yield displacements that
    do not correspond to the actual shift.

    :param image: The image as 2D numpy array.
    :return: The downsampled image, half the size in both dimensions.
    """
    blur = cv.GaussianBlur(image, (5, 5), 1)
    return blur[::2, ::2]


def align_pyramid(reference_channel: np.ndarray,
                  moving_channel: np.ndarray,
                  radius: int,
                  metric,
                  crop: float = 0.1,
                  levels: int = 5,
                  correction_radius: int = 3) -> tuple[int, int]:
    """
    Finds the best translation of one channel relative to another, coarse to fine.

    Recursively halves both channels until either the level budget is spent or the image
    is small enough, and runs an exhaustive search there. On the way back up, each
    estimate is doubled to account for the change in scale, the moving channel is
    shifted by it and only a small correction is searched for at the current resolution.
    This keeps the expensive full-radius search on the smallest image, so large
    displacements are found without the cost of searching for them directly.

    :param reference_channel: The channel to align against, as 2D array.
    :param moving_channel: The channel to be shifted, as 2D array.
    :param radius: The maximum displacement searched at the coarsest level.
    :param metric: A function taking two arrays and returning a score, where a
        higher value means a better match.
    :param crop: The fraction to remove per side before scoring, default 0.1.
    :param levels: The number of times the image may be halved, default 5.
    :param correction_radius: The displacement searched at each level on the way
        back up, default 3.
    :return: The total displacement as tuple (dy,dx).
    """
    if levels == 0 or reference_channel.shape[0] < 400:
        return align_single(reference_channel, moving_channel, radius, metric, crop)

    downsampled_reference = downsample(reference_channel)
    downsampled_moving = downsample(moving_channel)
    levels -= 1

    (dy, dx) = align_pyramid(downsampled_reference,
                             downsampled_moving, radius, metric, crop, levels, correction_radius)

    doubled_dy = dy * 2
    doubled_dx = dx * 2

    rolled = np.roll(moving_channel, (doubled_dy, doubled_dx), (0, 1))
    (correction_y, correction_x) = align_single(reference_channel,
                                                rolled, correction_radius, metric, crop)
    return doubled_dy + correction_y, doubled_dx + correction_x


def gradient_features(channel: np.ndarray) -> np.ndarray:
    """
    Computes the gradient magnitude of a channel.

    Aligning on gradients rather than raw intensities helps when the channels differ
    in brightness, as they do on plates where the filter absorbed much more light than
    on others. The absolute values then disagree between channels, but the position
    of the edges does not.

    :param channel: The channel as 2D array.
    :return: The gradient magniuted, same shape as the input.
    """
    gx = cv.Sobel(channel, cv.CV_64F, 1, 0, ksize=3)
    gy = cv.Sobel(channel, cv.CV_64F, 0, 1, ksize=3)

    return np.hypot(gx, gy)


def colorize(plate: np.ndarray,
             radius: int = 15,
             metric=l2,
             alignment_function=align_single,
             use_gradients: bool = False) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """
    Produces a color image from a single glass plate.

    Splits the plate into its three channels, aligns green and red to the blue channel,
    applies the resulting shifts and stacks the channels.

    :param plate: The full glass plate as 2D float array.
    :param radius: The maximum displacement searched in each direction, default 15.
    :param metric: A function taking two arrays and returning a score, where a
        higher value means a better match, default L2.
    :param alignment_function: The function used to find the displacement, either
        align_single or align_pyramid, default align_single.
    :param use_gradients: Whether to align on gradient magnitudes instead of raw
        intensities, default False. The shifts are always applied to the original channels.
    :return: A tuple of the color image, the green displacement (dy, dx) and the
        red displacement (dy, dx).
    """
    blue, green, red = split_channels(plate)

    blue_ref, green_ref, red_ref = blue, green, red
    if use_gradients:
        blue_ref = gradient_features(blue)
        green_ref = gradient_features(green)
        red_ref = gradient_features(red)

    dy_green, dx_green = alignment_function(blue_ref, green_ref, radius, metric)
    dy_red, dx_red = alignment_function(blue_ref, red_ref, radius, metric)

    green_aligned = np.roll(green, (dy_green, dx_green), (0, 1))
    red_aligned = np.roll(red, (dy_red, dx_red), (0, 1))

    stacked = stack_channels((blue, green_aligned, red_aligned))
    return stacked, (dy_green, dx_green), (dy_red, dx_red)


def save_img(image: np.ndarray, image_name: str) -> None:
    """
    Writes a float image to the disk as an 8-bit file.

    Values are clipped to [0,1] before scaling, since shifting channels can push
    individual pixels slightly outside the valid range.

    :param image: The image as numpy array with values in [0, 1], either a 2D grayscale
        array or a 3D BGR array of shape (height, width 3).
    :param image_name: The file path to write to, including the extension.
    """
    image = np.clip(image, 0, 1) * 255
    image = image.astype(np.uint8)
    cv.imwrite(image_name, image)


def main():
    image_list = Path("images").glob("*")
    Path("output").mkdir(exist_ok=True)

    with (open("output/offsets.txt", "w") as f):
        for file in image_list:
            image = to_float(read_img(str(file)))
            start = perf_counter()
            colorized_img, (dy_green, dx_green), (dy_red, dx_red) = colorize(
                image, metric=l2, alignment_function=align_pyramid, use_gradients=True)
            end = perf_counter()
            f.write(f"Image Name: {file}\n"
                    f"Green Offset (dy, dx): ({dy_green}, {dx_green})\n"
                    f"Red Offset (dy, dx): ({dy_red}, {dx_red})\n"
                    f"Elapsed Time: {end - start:.2f}s\n\n")

            save_img(colorized_img, f"output/{file.stem}.jpg")


if __name__ == "__main__":
    main()
