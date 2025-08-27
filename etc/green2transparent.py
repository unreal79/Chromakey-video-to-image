# -*- coding: utf-8 -*-

import os
import math
from PIL import Image
import numpy as np
import cv2


def split_video_into_frames(video_path, output_folder):
    """
    Splits a video into individual frames and saves them as image files.

    Args:
        video_path (str): The path to the input video file.
        output_folder (str): The directory where the extracted frames will be saved.
    """

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video was opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    frame_count = 0
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Break the loop if the video has ended
        if not ret:
            break

        # Construct the filename for the current frame
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.png")

        # Save the frame as an image file
        cv2.imwrite(frame_filename, frame)

        print(f"Saved {frame_filename}")
        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Video processing complete. {frame_count} frames extracted to {output_folder}")


def greenkey(rgba_img, sensitivity=0.8):
    sens = math.floor(255*sensitivity)

    arr = np.array(rgba_img)
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    a = arr[:, :, 3]
    cond = ((g < sens) | (g < b+5))
    arr[:, :, 3] = cond*255
    cond = g > b
    arr[:, :, 1] = cond*b + (~cond)*g # un/comment

    return Image.fromarray(arr)


def main():
    img = Image.open('pic.png')
    rgba_img = img.convert('RGBA')
    out = greenkey(rgba_img, 0.8)
    out.save('changed.png', 'PNG')

    video_file = "sample_640x360.mp4"  # Replace with your video file name
    output_directory = "extracted_frames"
    split_video_into_frames(video_file, output_directory)

if __name__ == '__main__':
    main()
