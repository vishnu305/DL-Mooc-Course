import numpy as np
import cv2
import matplotlib.pyplot as plt

RGB= cv2.imread('RGB.jpg')
B,G,R= cv2.split(RGB)

cv2.imshow('Original Image', RGB)
cv2.waitKey(0)

# Visualize Blue, red and Green
# cv2.imshow('blue', B)
# cv2.waitKey(0)

# cv2.imshow('Red', R)
# cv2.waitKey(0)

# cv2.imshow('Green', G)
# cv2.waitKey(0)

#inverting colors
BGR= cv2.cvtColor(RGB, cv2.COLOR_BGR2RGB)
cv2.imshow('Reverse image',BGR)
cv2.waitKey(0)


cv2.destroyAllWindows()