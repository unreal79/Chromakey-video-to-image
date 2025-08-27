# -*- coding: utf-8 -*-

""" Chromakey-like video cropping without actual green screen. Resulted
    PNG files will contain alpha channel for transparency.

    How to use:
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
"""

import os
import multiprocessing
import signal
import cv2 # pip install opencv-python
import numpy as np


def thread_worker(img, frame_count, output_folder, mask_as_hsv, mask_channel, margin,
            kernel_ones, dilate, blur, mask_out):
    # Ignore SIGINT in the child process
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    if mask_as_hsv:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # change to HSV color space
        channels = cv2.split(hsv)
    else:
        channels = cv2.split(img)
    channel_mask = channels[mask_channel]

    # get uniques
    unique_colors, counts = np.unique(channel_mask, return_counts=True)

    # sort through and grab the most abundant unique color
    big_color = None
    biggest = -1
    for a in range(len(unique_colors)):
        if counts[a] > biggest:
            biggest = counts[a]
            big_color = int(unique_colors[a])

    # get the color mask
    mask = cv2.inRange(channel_mask, big_color - margin, big_color + margin)

    # smooth out the mask
    if dilate > 0 and kernel_ones > 0:
        kernel = np.ones((kernel_ones, kernel_ones), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations = dilate)
    if blur > 0:
        mask = cv2.medianBlur(mask, blur)

    crop = np.dstack((img, mask)) # add mask as alpha channel

    if mask_out: # to see what's masked out
        frame_filename = os.path.join(output_folder,
                f"{frame_count:05d}_{'HSV' if mask_as_hsv else 'BGR'}_{mask_channel}_mask.png")
        cv2.imwrite(frame_filename, crop)
        print(f"Saved: {frame_filename}")

    crop[:, :, 3] = cv2.bitwise_not(mask)

    frame_filename = os.path.join(output_folder,
            f"{frame_count:05d}_{'HSV' if mask_as_hsv else 'BGR'}_{mask_channel}.png")
    cv2.imwrite(frame_filename, crop)
    print(f"Saved: {frame_filename}")


def chromakey_video2png(path: str, output_folder: str, mask_as_hsv = True, mask_channel = 1, margin = 50,
                        kernel_ones = 3, dilate = 1, blur = 5, frames_max = -1, mask_out = False):
    """Splits video to PNG files and crops them.
    Args:
        path (str): Path to input video
        output_folder (str): Output folder for PNG files
        mask_as_hsv (bool, optional): Mask based on HSV color space? Defaults to True.
        mask_channel (int, optional): 0 = Hue/Blue, 1 = Saturation/Green, 2 = Value/Red. Defaults to 1.
        margin (int, optional): Crop margin. Defaults to 50.
        kernel_ones (int, optional): Crop blur kernel size. Defaults to 3.
        dilate (int, optional): Crop dilate size. Defaults to 1.
        blur (int, optional): Crop blur size. Defaults to 5.
        frames_max (int, optional): Maximum number of frames to process. Defaults to -1 (all files).
        mask_out (bool, optional): Additionally output masked-out file. Defaults to False.
    """
    cap = cv2.VideoCapture(path)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pool = multiprocessing.Pool(processes=(multiprocessing.cpu_count() - 1))

    frame_count = 0
    done = False
    try:
        while not done:
            ret, img = cap.read()
            if not ret:
                done = True
                continue

            pool.apply_async(
                thread_worker,
                args=(img, frame_count, output_folder, mask_as_hsv, mask_channel, margin,
                        kernel_ones, dilate, blur, mask_out)
            ).get(600) # timeout 600 secs to catch KeyboardInterrupt

            frame_count += 1
            if frame_count >= frames_max and frames_max > 0:
                done = True
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print("Main process caught KeyboardInterrupt.")
        pool.terminate()
    finally:
        cap.release()
        print("Job finished")


def find_best():
    """
    Creates a set of cropped images to choose best result.
    You have to decide what's a best option and use it on full video.
    """
    for ix in [True, False]:
        for jx in [0, 1, 2]:
            chromakey_video2png(
                path = "example/416530553804341248.mp4",
                output_folder = "example/find_best",
                mask_as_hsv = ix,
                mask_channel = jx,
                margin = 30,
                kernel_ones = 0,
                dilate = 0,
                blur = 0,
                frames_max = 1,
                mask_out = True
            )


def main():
    chromakey_video2png(
            path = "example/416530553804341248.mp4", # input video
            output_folder = "example/chromakey_out", # output directory
            mask_as_hsv = True, # convert to HSV color space to mask?
            mask_channel = 0, # 0 = Hue/Blue, 1 = Saturation/Green, 2 = Value/Red
            margin = 30, # other mask options
            kernel_ones = 3,
            dilate = 0,
            blur = 0,
            frames_max = -1, # if < 0 == all feames
            mask_out = False # additionally output masked-out file
    )


if __name__ == '__main__':
    find_best() # choose best result
    # main() # uncomment
