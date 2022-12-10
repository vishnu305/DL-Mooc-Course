import cv2

image1= cv2.imread("Cat.jpg",-1)

cv2.imshow("Original image", image1)
cv2.waitKey(0)

#setting the desired width and height
width=image1.shape[1]*50//100
height=image1.shape[0]*50//100

#creating new image matrix resized with different interpolation
dim=(width,height)
resized_NEAREST= cv2.resize(image1,dim,cv2.INTER_NEAREST)
resized_AREA= cv2.resize(image1,dim,cv2.INTER_AREA)
resized_LINEAR= cv2.resize(image1,dim,cv2.INTER_LINEAR)
resized_CUBIC= cv2.resize(image1,dim,cv2.INTER_CUBIC)


#visualizing resized image
cv2.imshow("RESIZED NEAREST INTERPOLATION", resized_NEAREST)
cv2.waitKey(0)

cv2.imshow("RESIZED AREA INTERPOLATION", resized_AREA)
cv2.waitKey(0)

cv2.imshow("RESIZED LINEAR INTERPOLATION", resized_LINEAR)
cv2.waitKey(0)

cv2.imshow("RESIZED CUBIC INTERPOLATION", resized_CUBIC)
cv2.waitKey(0)

cv2.destroyAllWindows()