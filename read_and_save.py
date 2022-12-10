import numpy as np
import cv2
cat=cv2.imread('cat.jpg',0)
cv2.imwrite('cat2.jpg',cat)

cv2.imshow('Cat Window',cat)

cv2.waitKey(0)
cv2.destroyAllWindows()