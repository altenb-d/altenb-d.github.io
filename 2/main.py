import os
import time

import cv2
import numpy as np
from scipy.signal import convolve2d
from align_image_code import align_images

D_x = np.array([[1, -1]])
D_y = np.array([[1], [-1]])


def load_image(filename: str, gray: bool = False) -> np.ndarray:
    if gray:
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(filename)
    return image.astype(np.float64) / 255


def save_image(filename: str, image: np.ndarray) -> None:
    os.makedirs("output images", exist_ok=True)
    image = np.clip(np.asarray(image, dtype=np.float64), 0, 1)
    cv2.imwrite(f"output images/{filename}", (image * 255).round().astype(np.uint8))


def rescale(image: np.ndarray) -> np.ndarray:
    return (image - image.min()) / (image.max() - image.min())


def crop_same_size(a: np.ndarray, b: np.ndarray) -> tuple:
    height = min(a.shape[0], b.shape[0])
    width = min(a.shape[1], b.shape[1])
    return a[:height, :width], b[:height, :width]


def zero_padding(image: np.ndarray, kernel_height: int, kernel_width: int) -> np.ndarray:
    """
    Adds zeros around the image so the output has the same size
    :param image: The image to add padding to.
    :param kernel_height: The height of the kernel.
    :param kernel_width: The width of the kernel.
    :return: The padded image.
    """
    top, left = kernel_height // 2, kernel_width // 2
    padded = np.zeros((image.shape[0] + kernel_height - 1, image.shape[1] + kernel_width - 1))
    padded[top:top + image.shape[0], left: left + image.shape[1]] = image
    return padded


def convolution_four_loops(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kernel = np.flip(kernel)
    kernel_height, kernel_width = kernel.shape
    padded = zero_padding(image, kernel_height, kernel_width)
    output = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result = 0
            for k in range(kernel_height):
                for l in range(kernel_width):
                    result += padded[i + k, j + l] * kernel[k, l]
            output[i, j] = result
    return output


def convolution_two_loops(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kernel = np.flip(kernel)
    kernel_height, kernel_width = kernel.shape
    padded = zero_padding(image, kernel_height, kernel_width)
    output = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i: i + kernel_height, j:j + kernel_width]
            output[i, j] = np.sum(window * kernel)
    return output


def part_1_1(selfie_filepath):
    image = load_image(selfie_filepath, True)
    image = cv2.resize(image, (200, int(200 * image.shape[0] / image.shape[1])))
    box = np.ones((9, 9)) / 81

    t = time.time()
    a = convolution_four_loops(image, box)
    t = time.time() - t
    print(f"1.1 with four loops: {t:.2f}s")

    t = time.time()
    b = convolution_two_loops(image, box)
    t = time.time() - t
    print(f"1.1 with two loops: {t:.2f}s")

    t = time.time()
    c = convolve2d(image, box, mode="same")
    t = time.time() - t
    print(f"1.1 with convolve2d: {t:.2f}s")

    print(f"1.1 max difference from convolve2d: {np.abs(a - c).max():.1e} and {np.abs(b - c).max():.1e}")

    save_image("1_1_selfie_gray.png", image)
    save_image("1_1_box.png", b)
    save_image("1_1_D_x.png", rescale(convolution_two_loops(image, D_x)))
    save_image("1_1_D_y.png", rescale(convolution_two_loops(image, D_y)))


def gradient_magnitude(image: np.ndarray):
    dx = convolve2d(image, D_x, mode="same", boundary="symm")
    dy = convolve2d(image, D_y, mode="same", boundary="symm")
    return dx, dy, np.sqrt(dx ** 2 + dy ** 2)


def part_1_2(cameraman_file: str, threshold: float = 0.2):
    image = load_image(cameraman_file, gray=True)
    dx, dy, magnitude = gradient_magnitude(image)
    save_image("1_2_dx.png", rescale(dx))
    save_image("1_2_dy.png", rescale(dy))
    save_image("1_2_magnitude.png", rescale(magnitude))
    save_image("1_2_edges.png", magnitude > threshold)

    for value in np.arange(0.1, 0.4, 0.05):
        save_image(f"1_2_edges_threshold{value:.2f}.png", magnitude > value)


def gaussian_kernel(sigma: float) -> np.ndarray:
    size = int(6 * sigma) + 1
    kernel = cv2.getGaussianKernel(size, sigma)
    return kernel @ kernel.T


def part_1_3(cameraman_file: str, sigma: float = 2, threshold: float = 0.05):
    image = load_image(cameraman_file, True)
    kernel = gaussian_kernel(sigma)

    blurred = convolve2d(image, kernel, mode="same", boundary="symm")
    _, _, magnitude_1 = gradient_magnitude(blurred)

    dog_x = convolve2d(kernel, D_x)
    dog_y = convolve2d(kernel, D_y)
    dx = convolve2d(image, dog_x, mode="same", boundary="symm")
    dy = convolve2d(image, dog_y, mode="same", boundary="symm")
    magnitude_2 = np.sqrt(dx ** 2 + dy ** 2)

    print(f"1.3 max difference between both ways: {np.abs(magnitude_1 - magnitude_2).max():.1e}")

    save_image("1_3_blurred.png", blurred)
    save_image("1_3_blur_edges.png", magnitude_1 > threshold)
    save_image("1_3_dog_x.png", rescale(dog_x))
    save_image("1_3_dog_y.png", rescale(dog_y))
    save_image("1_3_dog_edges.png", magnitude_2 > threshold)
    save_image("1_3_blur_magnitude.png", rescale(magnitude_1))
    save_image("1_3_dog_magnitude.png", rescale(magnitude_2))


def blur(image: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.filter2D(image, -1, gaussian_kernel(sigma), borderType=cv2.BORDER_REFLECT)


def sharpen(image: np.ndarray, sigma: float, alpha: float) -> np.ndarray:
    gaussian = gaussian_kernel(sigma)
    impulse = np.zeros_like(gaussian)
    impulse[gaussian.shape[0] // 2, gaussian.shape[1] // 2] = 1
    kernel = (1 + alpha) * impulse - alpha * gaussian
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)


def part_2_1(files: list, sharp_file: str, sigma: float = 2) -> None:
    for file in files:
        image = load_image(file)
        name = os.path.splitext(os.path.basename(file))[0]
        blurred = blur(image, sigma)
        save_image(f"2_1_{name}_blurred.png", blurred)
        save_image(f"2_1_{name}_details.png", rescale(image - blurred))
        for alpha in [0.5, 1, 2, 5]:
            save_image(f"2_1_{name}_alpha{alpha}.png", sharpen(image, sigma, alpha))

    image = load_image(sharp_file)
    blurred = blur(image, sigma)
    save_image("2_1_eval_blurred.png", blurred)
    save_image("2_1_eval_resharpened.png", sharpen(blurred, sigma, 2))


def align_pair(file_1: str, file_2: str, name: str) -> None:
    aligned_1, aligned_2 = align_images(load_image(file_1), load_image(file_2))
    save_image(f"2_2_{name}_aligned_1.png", aligned_1)
    save_image(f"2_2_{name}_aligned_2.png", aligned_2)


def hybrid_image(image_1: np.ndarray, image_2: np.ndarray, sigma_1: float, sigma_2: float) -> tuple:
    low = blur(image_1, sigma_1)
    high = image_2 - blur(image_2, sigma_2)
    return low, high, low + high


def save_fft(filename: str, image: np.ndarray) -> None:
    save_image(filename, rescale(np.log(np.abs(np.fft.fftshift(np.fft.fft2(image))) + 1e-8)))


def part_2_2(pairs: list) -> None:
    for i, (file_1, file_2, sigma_1, sigma_2) in enumerate(pairs):
        name = f"hybrid{i}"
        aligned_file_1 = f"output images/2_2_{name}_aligned_1.png"
        aligned_file_2 = f"output images/2_2_{name}_aligned_2.png"

        if not os.path.exists(aligned_file_1):
            align_pair(file_1, file_2, name)

        image_1 = load_image(aligned_file_1, gray=True)
        image_2 = load_image(aligned_file_2, gray=True)
        low, high, hybrid = hybrid_image(image_1, image_2, sigma_1, sigma_2)
        save_image(f"2_2_{name}.png", hybrid)

        if i == 0:
            save_image(f"2_2_low.png", low)
            save_image(f"2_2_high.png", rescale(high))
            save_fft("2_2_fft_image_1.png", image_1)
            save_fft("2_2_fft_image_2.png", image_2)
            save_fft("2_2_fft_low.png", low)
            save_fft("2_2_fft_high.png", high)
            save_fft("2_2_fft_hybrid.png", hybrid)


def gaussian_stack(image: np.ndarray, levels: int, sigma: float) -> list:
    return [image] + [blur(image, sigma * 2 ** k) for k in range(levels - 1)]


def laplacian_stack(image: np.ndarray, levels: int, sigma: float) -> list:
    gaussian = gaussian_stack(image, levels, sigma)
    return [gaussian[k] - gaussian[k + 1] for k in range(levels - 1)] + [gaussian[-1]]


def blend(image_1: np.ndarray, image_2: np.ndarray, mask: np.ndarray, levels: int, sigma: float) -> tuple:
    mask = np.dstack([mask] * 3)
    laplacian_1 = laplacian_stack(image_1, levels, sigma)
    laplacian_2 = laplacian_stack(image_2, levels, sigma)
    gaussian_mask = gaussian_stack(mask, levels, sigma)
    part_1 = [laplacian_1[k] * (1 - gaussian_mask[k]) for k in range(levels)]
    part_2 = [laplacian_2[k] * gaussian_mask[k] for k in range(levels)]
    blended = [part_1[k] + part_2[k] for k in range(levels)]
    return sum(blended), part_1, part_2, blended


def save_process(filename: str, part_1: list, part_2: list, blended: list) -> None:
    rows = [np.hstack([rescale(part_1[k]), rescale(part_2[k]), rescale(blended[k])]) for k in (0, 2, 4)]
    rows.append(np.hstack([sum(part_1), sum(part_2), sum(blended)]))
    save_image(filename, np.vstack(rows))


def part_2_3_and_2_4(apple_file: str, orange_file: str, blends: list, levels: int = 6, sigma: float = 2) -> None:
    apple, orange = crop_same_size(load_image(apple_file), load_image(orange_file))

    save_image("2_3_apple_gaussian.png", np.hstack(gaussian_stack(apple, levels, sigma)))
    save_image("2_3_apple_laplacian.png", np.hstack([rescale(l) for l in laplacian_stack(apple, levels, sigma)]))
    save_image("2_3_orange_gaussian.png", np.hstack(gaussian_stack(orange, levels, sigma)))
    save_image("2_3_orange_laplacian.png", np.hstack([rescale(l) for l in laplacian_stack(orange, levels, sigma)]))

    mask = np.zeros(apple.shape[:2])
    mask[:, apple.shape[1] // 2:] = 1
    oraple, part_1, part_2, blended = blend(apple, orange, mask, levels, sigma)
    save_image("2_4_oraple.png", oraple)
    save_process("2_3_figure_3_42.png", part_1, part_2, blended)

    for i, (file_1, file_2, mask_file) in enumerate(blends):
        image_1, image_2 = crop_same_size(load_image(file_1), load_image(file_2))
        mask = load_image(mask_file, gray=True)[:image_1.shape[0], :image_1.shape[1]]
        result, part_1, part_2, blended = blend(image_1, image_2, mask, levels, sigma)
        save_image(f"2_4_blend_{i}.png", result)
        save_process(f"2_4_blend_{i}_process.png", part_1, part_2, blended)


if __name__ == "__main__":
    part_1_1("input images/selfie.png")
    part_1_2("input images/cameraman.png")
    part_1_3("input images/cameraman.png")

    part_2_1(["input images/taj.jpg", "input images/blurry.jpeg"], "input images/selfie.png")

    part_2_2([
        ("input images/DerekPicture.jpg", "input images/nutmeg.jpg", 7, 4),
        ("input images/hybrid1_1.png", "input images/hybrid1_2.jpeg", 7, 2),
        ("input images/hybrid2_1.png", "input images/hybrid2_2.jpg", 3, 0.8)
    ])

    part_2_3_and_2_4("input images/apple.jpeg",
                     "input images/orange.jpeg",
                     blends=[
                        ("input images/blend1_1.jpg", "input images/blend1_2.jpg", "input images/blend1_mask.png"),
                        ("input images/blend2_1.jpg", "input images/blend2_2.jpg", "input images/blend2_mask.png")
                     ])
