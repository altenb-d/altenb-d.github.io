# CS 180 — Project 1: Colorizing the Prokudin-Gorskii Photo Collection

Aligns the three color channels of a digitized glass plate and writes out a
single color image.

## Requirements

Python 3.10 or newer (the code uses `tuple[...]` type hints).

```
pip install numpy opencv-python
```

## Folder layout

```
main.py
images/          input plates (.jpg and .tif), provided set
own_images/      input plates chosen from the LoC collection
output/          created automatically, holds results + offsets.txt
```

The input folders must contain only plate images. Anything else in there will be
read as a plate and will either fail or produce nonsense.

## Running

```
python main.py
```

`main()` iterates over every file in the input folder, writes one JPG per plate
to the output folder and records the displacements and timings in
`output/offsets.txt`.

To point the script at a different set of plates, change the
input and output paths inside `main()`.

## Configuration

The configuration used for the final results is set in the `colorize()` call
inside `main()`:

```
colorize(image, metric=l2, alignment_function=align_pyramid, use_gradients=True)
```

Each argument can be swapped independently:

| Argument             | Options                         | Meaning                                                   |
|----------------------|---------------------------------|-----------------------------------------------------------|
| `metric`             | `l2`, `ncc`                     | Scoring function. Both return higher-is-better.           |
| `alignment_function` | `align_single`, `align_pyramid` | Exhaustive search at full resolution, or coarse-to-fine.  |
| `use_gradients`      | `False`, `True`                 | Score on raw intensities or on Sobel gradient magnitudes. |

`align_single` is only practical on the small JPEGs; on the full-resolution
TIFFs the displacements exceed the search radius.

The remaining parameters (`radius`, `crop`, `levels`, `correction_radius`) have
defaults that were used unchanged for every plate. They can be overridden per
call if needed.

## Reproducing the comparison runs on the webpage

| Webpage section            | Call                                                                |
|----------------------------|---------------------------------------------------------------------|
| Pyramid, raw pixels        | `metric=l2, alignment_function=align_pyramid, use_gradients=False`  |
| Pyramid, NCC on raw pixels | `metric=ncc, alignment_function=align_pyramid, use_gradients=False` |
| Final configuration        | `metric=l2, alignment_function=align_pyramid, use_gradients=True`   |

Each run overwrites the output folder, so rename it between runs if you want to
keep the results side by side.

## Runtime

Roughly 9–14 seconds per full-resolution TIFF and under a second per JPEG on a
normal laptop. Using NCC instead of L2 roughly triples this.