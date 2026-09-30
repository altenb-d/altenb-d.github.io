CS180 Project 2: Fun with Filters and Frequencies
=================================================

Files
-----
main.py               All code for Parts 1.1 - 2.4
align_image_code.py   Provided starter code for aligning the hybrid image pairs
input images/         All input images and blend masks

Requirements
------------
pip install numpy scipy opencv-python scikit-image matplotlib

How to run
----------
From inside the code/ folder:

    python main.py

All results are written to "output images/" (created automatically).
The file names start with the part they belong to, e.g. 1_2_edges.png or
2_4_oraple.png.

Hybrid images (Part 2.2)
------------------------
On the first run, a window opens for every hybrid pair. Click two points in
the first image (e.g. left eye, then right eye), then the same two points in
the second image. The aligned images are saved as
"output images/2_2_hybrid<i>_aligned_1.png" / "_aligned_2.png" and reused on
later runs, so you only have to click once. Delete these files to align a
pair again.

If the click window does not open (e.g. inside an IDE that shows plots in its
own tool window), add these lines at the very top of align_image_code.py:

    import matplotlib
    matplotlib.use("TkAgg")

Where the parts are
-------------------
Part 1.1   zero_padding, convolution_four_loops, convolution_two_loops, part_1_1
Part 1.2   gradient_magnitude, part_1_2
Part 1.3   gaussian_kernel, part_1_3
Part 2.1   blur, sharpen, part_2_1
Part 2.2   align_pair, hybrid_image, save_fft, part_2_2
Part 2.3   gaussian_stack, laplacian_stack
Part 2.4   blend, save_process, part_2_3_and_2_4

Parameters used for the results on the webpage
----------------------------------------------
1.2   threshold 0.2
1.3   sigma 2, threshold 0.05
2.1   sigma 2, alpha 0.5 / 1 / 2 / 5, evaluation with alpha 2
2.2   Derek + Nutmeg: sigma_1 7, sigma_2 4
me + koala:     sigma_1 7, sigma_2 2
me + pixel art: sigma_1 3, sigma_2 0.8
2.3   6 levels, sigma 2 (doubling each level)

Notes
-----
The blend inputs in "input images" (blend1_*, blend2_*) were prepared
beforehand: the apple was scaled to the tomato's size and the eye photo was
moved onto the palm, and the masks were drawn to match.