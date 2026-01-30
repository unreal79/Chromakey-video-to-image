# -*- coding: utf-8 -*-

""" Chromakey-like video/image processing without actual green screen.
    Automatically removes backgrounds and outputs PNG files with alpha channel for transparency.

    This tool supports both video files and static images, processing them to remove
    backgrounds based on the most dominant color in the image/frame.

    Usage via command line:

    Process a video:
        python chromakey_video2png.py --video input.mp4 -o output/

    Process an image:
        python chromakey_video2png.py --image input.png -o output/

    Find best parameters (outputs 6 test combinations):
        python chromakey_video2png.py --find-best input.mp4 -o test_results/
        python chromakey_video2png.py --find-best-image input.png -o test_results/

    Workflow for optimal results:
    1. Run with --find-best or --find-best-image to generate test outputs
    2. Review the 6 generated images to find the best result
       (e.g., if "HSV_0.png" looks best, use --hsv --mask-channel 0)
    3. Process with those parameters, adjusting --margin, --kernel, --dilate,
       and --blur as needed for fine-tuning
    4. For videos, test with -f 10 (first 10 frames) before processing all frames

    For full options, run: python chromakey_video2png.py --help

    Utilizes OpenCV and multiprocessing for maximum performance.
    Based on https://stackoverflow.com/a/66355953
"""

import os
import argparse
import multiprocessing
import signal
import cv2 # pip install opencv-python
import numpy as np


# Degree of desaturation: 1.0 = full grayscale, 0.0 = no change
DESATURATE_FACTOR = 1.0


def post_process_image(img_rgba, mask_as_hsv, mask_channel, big_color, margin):
    """
    Desaturates pixels that are close to the background color and adjacent to transparency.
    """
    img_bgr = img_rgba[:, :, :3]
    alpha = img_rgba[:, :, 3]

    if mask_as_hsv:
        # HSV works better for colored backgrounds
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        target_channel = hsv[:, :, mask_channel]
    else:
        # BGR channels
        target_channel = img_bgr[:, :, mask_channel]

    # 1. Identify pixels close to background color in target channel
    color_mask = cv2.inRange(target_channel, big_color - margin, big_color + margin)

    # 2. Identify pixels adjacent to fully transparent pixels (alpha == 0)
    # We use a slightly larger kernel or iterations to catch the halo better
    transparent_mask = (alpha == 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    near_transparent = cv2.dilate(transparent_mask, kernel, iterations=2)

    # 3. Final mask: near-transparent, matches color, but is NOT transparent itself
    fix_mask = (color_mask > 0) & (near_transparent > 0) & (alpha > 0)

    # 4. Apply desaturation
    pixels_affected = np.sum(fix_mask)
    if pixels_affected > 0:
        # Convert to Gray using weighted sum
        # Calculate gray values as float for factor blending
        gray = (0.299 * img_bgr[:, :, 2] + 0.587 * img_bgr[:, :, 1] + 0.114 * img_bgr[:, :, 0])

        for i in range(3): # B, G, R channels
            original = img_bgr[fix_mask, i].astype(float)
            target = gray[fix_mask].astype(float)
            # Interpolate: original -> target based on DESATURATE_FACTOR
            img_bgr[fix_mask, i] = (original + (target - original) * DESATURATE_FACTOR).astype(np.uint8)

    print(f"Post-process: affected {pixels_affected} pixels (mask_channel={mask_channel}, color={big_color}, margin={margin})")
    return img_rgba


def thread_worker(img, frame_count, output_folder, mask_as_hsv, mask_channel, margin,
            kernel_ones, dilate, blur, mask_out, post_process, post_process_margin):
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
    big_color = 0
    biggest = -1
    for a in range(len(unique_colors)):
        if counts[a] > biggest:
            biggest = counts[a]
            big_color = int(unique_colors[a])

    # get the color mask (clamp bounds to valid uint8 range)
    lower = int(max(0, big_color - margin))
    upper = int(min(255, big_color + margin))
    mask = cv2.inRange(channel_mask, lower, upper)  # type: ignore

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

    # Set alpha channel
    alpha = cv2.bitwise_not(mask)
    crop[:, :, 3] = alpha

    # Apply post-processing (desaturate edges)
    if post_process:
        crop = post_process_image(crop, mask_as_hsv, mask_channel, big_color, post_process_margin)

    frame_filename = os.path.join(output_folder,
            f"{frame_count:05d}_{'HSV' if mask_as_hsv else 'BGR'}_{mask_channel}.png")
    cv2.imwrite(frame_filename, crop)
    print(f"Saved: {frame_filename}")


def chromakey_image2png(path: str, output_folder: str, output_filename: str = "", mask_as_hsv = True,
                        mask_channel = 1, margin = 50, kernel_ones = 3, dilate = 1, blur = 5,
                        mask_out = False, post_process = False, post_process_margin = 20):
    """Processes a single image and applies chromakey effect.
    Args:
        path (str): Path to input image
        output_folder (str): Output folder for PNG file
        output_filename (str, optional): Output filename. If None, uses input filename with suffix.
        mask_as_hsv (bool, optional): Mask based on HSV color space? Defaults to True.
        mask_channel (int, optional): 0 = Hue/Blue, 1 = Saturation/Green, 2 = Value/Red. Defaults to 1.
        margin (int, optional): Crop margin. Defaults to 50.
        kernel_ones (int, optional): Crop blur kernel size. Defaults to 3.
        dilate (int, optional): Crop dilate size. Defaults to 1.
        blur (int, optional): Crop blur size. Defaults to 5.
        mask_out (bool, optional): Additionally output masked-out file. Defaults to False.
        post_process (bool, optional): Desaturate edges. Defaults to False.
        post_process_margin (int, optional): Margin for post-processing. Defaults to 20.
    """
    img = cv2.imread(path) # open image file

    if img is None:
        print(f"Error: Could not load image from {path}")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Determine output filename prefix
    if output_filename is None:
        base_name = os.path.splitext(os.path.basename(path))[0]
        output_filename = base_name

    try:
        # Process image directly without multiprocessing pool (it's only one image)
        if mask_as_hsv:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            channels = cv2.split(hsv)
        else:
            channels = cv2.split(img)
        channel_mask = channels[mask_channel]

        # get uniques
        unique_colors, counts = np.unique(channel_mask, return_counts=True)

        # sort through and grab the most abundant unique color
        big_color = 0
        biggest = -1
        for a in range(len(unique_colors)):
            if counts[a] > biggest:
                biggest = counts[a]
                big_color = int(unique_colors[a])

        # get the color mask (clamp bounds to valid uint8 range)
        lower = int(max(0, big_color - margin))
        upper = int(min(255, big_color + margin))
        mask = cv2.inRange(channel_mask, lower, upper)  # type: ignore

        # smooth out the mask
        if dilate > 0 and kernel_ones > 0:
            kernel = np.ones((kernel_ones, kernel_ones), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations = dilate)
        if blur > 0:
            mask = cv2.medianBlur(mask, blur)

        crop = np.dstack((img, mask)) # add mask as alpha channel

        if mask_out: # to see what's masked out
            frame_filename = os.path.join(output_folder,
                    f"{output_filename}_{'HSV' if mask_as_hsv else 'BGR'}_{mask_channel}_mask.png")
            cv2.imwrite(frame_filename, crop)
            print(f"Saved: {frame_filename}")

        # Set alpha channel
        alpha = cv2.bitwise_not(mask)
        crop[:, :, 3] = alpha

        # Apply post-processing (desaturate edges)
        if post_process:
            crop = post_process_image(crop, mask_as_hsv, mask_channel, big_color, post_process_margin)

        frame_filename = os.path.join(output_folder,
                f"{output_filename}_{'HSV' if mask_as_hsv else 'BGR'}_{mask_channel}.png")
        cv2.imwrite(frame_filename, crop)
        print(f"Saved: {frame_filename}")
    except KeyboardInterrupt:
        print("Process caught KeyboardInterrupt.")
    finally:
        print("Job finished")


def chromakey_video2png(path: str, output_folder: str, mask_as_hsv = True, mask_channel = 1, margin = 50,
                        kernel_ones = 3, dilate = 1, blur = 5, frames_max = -1, mask_out = False,
                        post_process = False, post_process_margin = 20):
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
        post_process (bool, optional): Desaturate edges. Defaults to False.
        post_process_margin (int, optional): Margin for post-processing. Defaults to 20.
    """
    cap = cv2.VideoCapture(path) # open video file

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
                        kernel_ones, dilate, blur, mask_out, post_process, post_process_margin)
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


def find_best(path: str, output_folder: str, is_image: bool = False):
    """
    Creates a set of cropped images/frames to choose best result.
    You have to decide what's a best option and use it for full processing.

    Args:
        path (str): Path to input video or image
        output_folder (str): Output folder for the results
        is_image (bool, optional): If True, processes as image; if False, processes as video. Defaults to False.
    """
    for ix in [True, False]:
        for jx in [0, 1, 2]:
            if is_image:
                chromakey_image2png(
                    path = path,
                    output_folder = output_folder,
                    mask_as_hsv = ix,
                    mask_channel = jx,
                    margin = 30,
                    kernel_ones = 0,
                    dilate = 0,
                    blur = 0,
                    mask_out = True,
                    post_process = False,
                    post_process_margin = 110
                )
            else:
                chromakey_video2png(
                    path = path,
                    output_folder = output_folder,
                    mask_as_hsv = ix,
                    mask_channel = jx,
                    margin = 30,
                    kernel_ones = 0,
                    dilate = 0,
                    blur = 0,
                    frames_max = 1,
                    mask_out = True,
                    post_process = False,
                    post_process_margin = 110
                )


def main():
    parser = argparse.ArgumentParser(
        description='Chromakey-like video/image cropping without actual green screen. '
                    'Resulted PNG files will contain alpha channel for transparency.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Examples:
  Process a video:
    python chromakey_video2png.py --video input.mp4 -o output/ --margin 50 --blur 5

  Process an image:
    python chromakey_video2png.py --image input.png -o output/ --mask-channel 0

  Find best parameters:
    python chromakey_video2png.py --find-best input.mp4 -o best_results/
    python chromakey_video2png.py --find-best-image input.png -o best_results/
        '''
    )

    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-v', '--video', type=str,
                            help='Path to input video file')
    input_group.add_argument('-i', '--image', type=str,
                            help='Path to input image file')
    input_group.add_argument('--find-best', type=str,
                            help='Find best parameters for a video (outputs 6 combinations)')
    input_group.add_argument('--find-best-image', type=str,
                            help='Find best parameters for an image (outputs 6 combinations)')

    # Output options
    parser.add_argument('-o', '--output', type=str, required=True,
                       help='Output folder for PNG files')
    parser.add_argument('-f', '--frames', type=int, default=-1,
                       help='Maximum number of frames to process for videos. -1 = all frames (default: -1)')

    # Processing options
    color_space_group = parser.add_mutually_exclusive_group()
    color_space_group.add_argument('--hsv', action='store_true', default=True,
                       help='Use HSV color space for masking (default: True)')
    color_space_group.add_argument('--bgr', dest='hsv', action='store_false',
                       help='Use BGR color space for masking instead of HSV')
    parser.add_argument('-c', '--mask-channel', type=int, default=1,
                       choices=[0, 1, 2],
                       help='Mask channel: 0=Hue/Blue, 1=Saturation/Green, 2=Value/Red (default: 1)')
    parser.add_argument('-m', '--margin', type=int, default=50,
                       help='Crop margin (default: 50)')
    parser.add_argument('-k', '--kernel', type=int, default=3,
                       help='Crop blur kernel size (default: 3)')
    parser.add_argument('-d', '--dilate', type=int, default=1,
                       help='Crop dilate size (default: 1)')
    parser.add_argument('-b', '--blur', type=int, default=5,
                       help='Crop blur size (default: 5)')
    parser.add_argument('--mask-out', action='store_true',
                       help='Additionally output masked-out file')
    parser.add_argument('--post-process', action='store_true',
                       help='Apply post-processing (desaturate edges)')
    parser.add_argument('--post-margin', type=int, default=20,
                       help='Margin for post-processing (default: 20)')

    args = parser.parse_args()

    # Handle find-best commands
    if args.find_best:
        print(f"Finding best parameters for video: {args.find_best}")
        find_best(args.find_best, args.output, is_image=False)
        return

    if args.find_best_image:
        print(f"Finding best parameters for image: {args.find_best_image}")
        find_best(args.find_best_image, args.output, is_image=True)
        return

    # Process video or image
    if args.video:
        print(f"Processing video: {args.video}")
        chromakey_video2png(
            path=args.video,
            output_folder=args.output,
            mask_as_hsv=args.hsv,
            mask_channel=args.mask_channel,
            margin=args.margin,
            kernel_ones=args.kernel,
            dilate=args.dilate,
            blur=args.blur,
            frames_max=args.frames,
            mask_out=args.mask_out,
            post_process=args.post_process,
            post_process_margin=args.post_margin
        )
    elif args.image:
        print(f"Processing image: {args.image}")
        chromakey_image2png(
            path=args.image,
            output_folder=args.output,
            mask_as_hsv=args.hsv,
            mask_channel=args.mask_channel,
            margin=args.margin,
            kernel_ones=args.kernel,
            dilate=args.dilate,
            blur=args.blur,
            mask_out=args.mask_out,
            post_process=args.post_process,
            post_process_margin=args.post_margin
        )


if __name__ == '__main__':
    main()
