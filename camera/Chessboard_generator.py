import cv2
import numpy as np

width = 450
height = 350
chess_height = 50
chess_width = 25

image = np.zeros((width,height),dtype = np.uint8)
print(image.shape[0],image.shape[1])

for j in range(height):
    for i in range(width):
        if((int)(i/chess_height) + (int)(j/chess_width))%2:
            image[i,j] = 255;
cv2.imwrite("pic/chess.jpg",image)
cv2.imshow("chess",image)
cv2.waitKey(0)