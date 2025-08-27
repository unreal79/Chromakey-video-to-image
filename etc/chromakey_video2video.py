# -*- coding: utf-8 -*-

"""    Based on https://stackoverflow.com/a/66355953
"""

import os
import cv2 # pip install opencv-python
import numpy as np
from PIL import Image


# https://stackoverflow.com/a/66355953 -- the origibal code
def chromakey_video2video(path, out_path):

    # open up video
    cap = cv2.VideoCapture(path)

    scale = 0.5

    # grab one frame
    _, frame = cap.read()
    h, w = frame.shape[:2]
    h = int(h*scale)
    w = int(w*scale)
    res = (w, h)
    cap.release() # to reopen video, else first frame is lost

    cap = cv2.VideoCapture(path)
    # videowriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_path, fourcc, 30000/1001, res)

    # loop
    done = False
    while not done:
        # get frame
        ret, img = cap.read()
        if not ret:
            done = True
            continue

        # resize
        img = cv2.resize(img, res)

        # change to hsv
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)

        # get uniques
        unique_colors, counts = np.unique(h, return_counts=True)

        # sort through and grab the most abundant unique color
        big_color = None
        biggest = -1
        for a in range(len(unique_colors)):
            if counts[a] > biggest:
                biggest = counts[a]
                big_color = int(unique_colors[a])

        # get the color mask
        margin = 50
        mask = cv2.inRange(h, big_color - margin, big_color + margin)

        # smooth out the mask and invert
        kernel = np.ones((3,3), np.uint8)
        dilate = 1
        mask = cv2.dilate(mask, kernel, iterations = dilate) #
        blur = 5
        mask = cv2.medianBlur(mask, blur)
        mask = cv2.bitwise_not(mask)

        # crop out the image
        crop = np.zeros_like(img)
        crop[mask == 255] = img[mask == 255]

        # show
        cv2.imshow("Mask", mask)
        cv2.imshow("Blank", crop)
        cv2.imshow("Image", img)
        done = cv2.waitKey(1) == ord('q')

        # save
        out.write(crop)

    # close caps
    cap.release()
    out.release()


def main():
    chromakey_video2video("../example/416530553804341248.mp4", '../example/out_vid.avi')


if __name__ == '__main__':
    main() # uncomment
