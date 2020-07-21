import cv2
import numpy as np


image = cv2.imread('t1.jpg', 0)
image2 = cv2.imread('t1.jpg')
image = cv2.resize(image, (500, 500))
image2 = cv2.resize(image2, (500, 500))
cv2.waitKey(0)


ret, thresh_basic = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
cv2.imshow("Thresh basic", thresh_basic)

# thresh_addapt = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
# cv2.imshow("Thresh Adapt", thresh_addapt)

# Taking a matrix of size 5 as the kernel
kernel = np.ones((5, 5), np.uint8)

img_erosion = cv2.erode(thresh_basic, kernel, iterations=1)

#####################

ret, thresh_inv = cv2.threshold(img_erosion, 100, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("INV", thresh_inv)
#####################


# Find Canny edges

edged = cv2.Canny(img_erosion, 30, 200)
cv2.waitKey(0)

contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

cv2.imshow('Canny', edged)
cv2.waitKey(0)

# print("Number of Contours found = " + str(len(contours)))
cv2.imshow('Original', image2)
cv2.drawContours(image2, contours, -1, (0, 255, 0), 3)


cv2.imshow('Contours', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
