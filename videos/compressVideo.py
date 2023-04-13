import cv2
import numpy as np

cap = cv2.VideoCapture('videos/mon.avi')

codec = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('morning.mp4',codec,  30, (1920 ,1080 ) )

while True:
    ret, frame = cap.read()
    if ret == True:
        out.write(frame)
        #print(frame.shape)
    else:
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()