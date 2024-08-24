import cv2
import numpy as np


cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)


while True:

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    frame = np.concatenate([frame1, frame2])
    cv2.rectangle(frame1,(0,frame1.shape[0]//2-10), (frame1.shape[1], frame1.shape[0]//2+10), (0,0,0), 3, cv2.LINE_AA)
    cv2.rectangle(frame1,(frame1.shape[1]//2-10,0), (frame1.shape[1]//2+10,frame1.shape[0]), (0,0,0), 3, cv2.LINE_AA)
    cv2.rectangle(frame2,(0,frame2.shape[0]//2-10), (frame2.shape[1], frame2.shape[0]//2+10), (0,0,0), 3, cv2.LINE_AA)
    cv2.rectangle(frame2,(frame2.shape[1]//2-10,0), (frame2.shape[1]//2+10,frame2.shape[0]), (0,0,0), 3, cv2.LINE_AA)
        
    if (not ret1) or (not ret2):
        break
    cv2.imshow("cam1", frame1)
    cv2.imshow("YOLOv8 Detection", frame2)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break
cap1.release()
cap2.release()
cv2.destroyAllWindows()
