# Chromakey-video-to-image
Chromakey-like video cropping without actual green screen. Resulted PNG files will contain alpha channel for transparency.

## How to use:

Since there are a lot of different images types that need to be
cropped, it would be your job to choose which options give better results.
To do so:
1. Run `find_best()`, set `path` to your file. It will output 6 pairs of PNG
    files into `find_best` folder.
2. Choose best result (for example `00000_HSV_0.png` -- this means that
    HSV color space should be used (`mask_as_hsv = True`) and mask
    channel should be 0 (`mask_channel = 0`))
3. Run `chromakey_video2png()` with options from step 2. Preferably
    set `frames_max = 10` to proceed only first 10 frames.
4. Modify `margin` option in range from 10 to 90 to get best results.
5. Modify `kernel_ones`, `dilate` and `blur` options in range from 0 to 5
    to get best results.
6. Finally run `chromakey_video2png()` with `frames_max = -1` to
    process all frames.

Utilises `OpenCV` and multithreading (`multiprocessing`) for
maximum speed.

Based on https://stackoverflow.com/a/66355953
