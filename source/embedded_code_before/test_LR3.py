import threading
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from gpiozero import PWMOutputDevice, DigitalOutputDevice
import os
import time

os.environ['QT_QPA_PLATFORM'] = 'xcb'

PWMA = 13
AIN1 = 19
AIN2 = 26

PWMB = 12
BIN1 = 20
BIN2 = 16

L_Motor = PWMOutputDevice(PWMA)
R_Motor = PWMOutputDevice(PWMB)

AIN1_pin = DigitalOutputDevice(AIN1)
AIN2_pin = DigitalOutputDevice(AIN2)
BIN1_pin = DigitalOutputDevice(BIN1)
BIN2_pin = DigitalOutputDevice(BIN2)

def motor_back(speed):
    L_Motor.value = speed
    AIN2_pin.off()
    AIN1_pin.on()
    R_Motor.value = speed
    BIN2_pin.off()
    BIN1_pin.on()
    
def motor_go(speed):
    L_Motor.value = speed
    AIN2_pin.on()
    AIN1_pin.off()
    R_Motor.value = speed
    BIN2_pin.on()
    BIN1_pin.off()

def motor_stop():
    L_Motor.value = 0
    R_Motor.value = 0

def motor_right(speed):
    L_Motor.value = speed
    AIN2_pin.on()
    AIN1_pin.off()
    R_Motor.value = 0

def motor_left(speed):
    L_Motor.value = 0
    R_Motor.value = speed
    BIN2_pin.on()
    BIN1_pin.off()

speedSet = 1.0

classNames = {0: 'background',
              1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
              7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
              13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
              18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
              24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
              32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
              37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
              41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
              46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
              51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
              56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
              61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
              67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
              75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
              80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
              86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}


def id_class_name(class_id, classes):
    for key, value in classes.items():
        if class_id == key:
            return value
        
def img_preprocess(image):
    height, _, _ = image.shape
    image = image[int(height/2):,:,:]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image = cv2.resize(image, (200,66))
    image = cv2.GaussianBlur(image,(5,5),0)
    _, image = cv2.threshold(image,160,255,cv2.THRESH_BINARY_INV)
    image = image / 255
    return image

camera = cv2.VideoCapture(0)
camera2 = cv2.VideoCapture(2)
camera.set(3, 640)
camera.set(4, 480)
camera2.set(3, 640)
camera2.set(4, 480)

image = None
image2 = None
image_ok = 0

box_size = 0
carState = "stop"
running = True

imagednn = None

def capture_and_preprocess():
    global image, image2, image_ok, imagednn
    while running:
        _, image = camera.read()
        _, image2 = camera2.read()
        imagednn = image.copy()
        image_ok = 1

def opencvdnn_thread():
    global image_ok, box_size, carState, running, imagednn

    model = cv2.dnn.readNetFromTensorflow('/home/test/Downloads/AI_CAR/OpencvDnn/models/frozen_inference_graph.pb',
                                          '/home/test/Downloads/AI_CAR/OpencvDnn/models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')
    while running:
        if image_ok:
            image_height, image_width, _ = imagednn.shape
            
            model.setInput(cv2.dnn.blobFromImage(imagednn, size=(250, 250), swapRB=True))
            output = model.forward()
            
            for detection in output[0, 0, :, :]:
                confidence = detection[2]
                if confidence > .5:
                    class_id = detection[1]
                    class_name = id_class_name(class_id, classNames)
                    if class_name == "person":
                        box_x = detection[3] * image_width
                        box_y = detection[4] * image_height
                        box_width = detection[5] * image_width
                        box_height = detection[6] * image_height
                        box_size = box_width * box_height
                        
                        print("auto stop")
                
                        cv2.rectangle(imagednn, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (23, 230, 210), thickness=1)
                        cv2.putText(imagednn, class_name, (int(box_x), int(box_y + .05 * image_height)), cv2.FONT_HERSHEY_SIMPLEX, (.005 * image_width), (0, 0, 255))
            image_ok = 0

def main():
    global carState, running, imagednn, image2

    model_path = '/home/test/Downloads/model_0517_2.h5'
    model = load_model(model_path)
    
    try:
        while running:
            if image2 is not None:
                preprocessed = img_preprocess(image2)
                X = np.asarray([preprocessed])
                steering_angle = int(model.predict(X)[0])
                print("predict angle:", steering_angle)
                if carState == "go":
                    if 70 <= steering_angle <= 110:
                        print("go")
                        motor_go(speedSet)
                    elif steering_angle > 111:
                        print("right")
                        motor_right(speedSet)
                    elif steering_angle < 71:
                        print("left")
                        motor_left(speedSet)
                elif carState == "stop":
                    motor_stop()

                #cv2.imshow('pre', preprocessed)
                cv2.imshow('imagednn', imagednn)
                
                keyValue = cv2.waitKey(1)
                if keyValue == ord('q'):
                    running = False
                    break
                elif keyValue == 82:
                    print("go")
                    carState = "go"
                elif keyValue == 84:
                    print("stop")
                    carState = "stop"
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        running = False

if __name__ == '__main__':
    capture_thread = threading.Thread(target=capture_and_preprocess)
    dnn_thread = threading.Thread(target=opencvdnn_thread)

    capture_thread.start()
    dnn_thread.start()

    main()

    capture_thread.join()
    dnn_thread.join()
    camera.release()
    camera2.release()
    cv2.destroyAllWindows()
