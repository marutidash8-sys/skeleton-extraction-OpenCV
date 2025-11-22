import cv2
import numpy as np

img = cv2.imread('input/sample.jpg', 0)
_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

size = np.size(img)
skel = np.zeros(img.shape, np.uint8)

element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

done = False

while not done:
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
    temp = cv2.subtract(img, opened)
    skel = cv2.bitwise_or(skel, temp)
    img = cv2.erode(img, element)

    zeros = size - cv2.countNonZero(img)
    if zeros == size:
        done = True

cv2.imwrite("output/skeleton_output.png", skel)
print("Skeleton saved successfully!")
