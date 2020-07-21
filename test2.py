import cv2
import numpy as np
from matplotlib import pyplot as plt

image2 = cv2.imread('images/x4.jpeg')
image2 = cv2.resize(image2, (500,500))
img = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

ret, thresh_basic = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
# show thresholded image - debugging
cv2.imshow("Thresh basic", thresh_basic)

img = cv2.GaussianBlur(img, (5,5), 0)

edges = cv2.Canny(img,100,70)

cv2.imshow('edges',edges)

contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# get total countors
print("Number of Contours found = " + str(len(contours)))
if len(contours) > 1:
    print("MARKINGS DETECTED")

# show original img
cv2.imshow('Original', image2)
# draw contours on original img
cv2.drawContours(image2, contours, -1, (0, 255, 0), 1)

# show markings highlighted img
cv2.imshow('Contours', image2)
cv2.imwrite("Output Images/image2.jpg", image2)
cv2.waitKey(0)
cv2.destroyAllWindows()



plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()