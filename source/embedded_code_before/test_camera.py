import cv2
from ultralytics import YOLO
import numpy as np

ncnn_model = YOLO("/home/test/model-test/models/ciga_changseop_ncnn_model")

cap = cv2.VideoCapture(0)
#cap.set(3,480)
#cap.set(4,360)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    results = ncnn_model(frame)

    annotated_frame = frame

    for result in results:
        try:
            print(result.boxes.conf[result.boxes.conf>0.5])
            annotated_frame = result.plot()
        except Exception as e:
            print(f"Error plotting results: {e}")

    cv2.imshow("YOLOv8 Detection", annotated_frame)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()