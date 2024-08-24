import cv2
import numpy as np


cap = cv2.VideoCapture(0)
# cap.set(3,160)
# cap.set(4,120)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("YOLOv8 Detection", frame)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()