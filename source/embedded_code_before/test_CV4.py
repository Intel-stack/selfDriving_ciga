import threading
import time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from gpiozero import PWMOutputDevice, DigitalOutputDevice, Motor

# Define GPIO pins
PWMA = 12
AIN1 = 16
AIN2 = 20

PWMB = 13
BIN1 = 19
BIN2 = 26

# Initialize motors and pins
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
    R_Motor.value = speed
    BIN2_pin.off()
    BIN1_pin.on()

def motor_left(speed):
    L_Motor.value = speed
    AIN2_pin.off()
    AIN1_pin.on()
    R_Motor.value = speed
    BIN2_pin.on()
    BIN1_pin.off()

# Pretrained classes in the model
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
    return classes.get(class_id, 'Unknown')

def img_preprocess(image):
    height, _, _ = image.shape
    image = image[int(height/2):,:,:]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image = cv2.resize(image, (200,66))
    image = cv2.GaussianBlur(image,(5,5),0)
    _,image = cv2.threshold(image,160,255,cv2.THRESH_BINARY_INV)
    image = image / 255
    return image

camera = cv2.VideoCapture(0)
camera.set(3, 640)
camera.set(4, 480)

_, image = camera.read()
image_ok = False

box_size = 0
carState = "stop"
speedSet = 1.0

def opencvdnn_thread():
    global image, image_ok, box_size, carState
    model1 = cv2.dnn.readNetFromTensorflow('/home/test/Downloads/AI_CAR/OpencvDnn/models/frozen_inference_graph.pb',
                                           '/home/test/Downloads/AI_CAR/OpencvDnn/models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

    while True:
        if image_ok:
            imagednn = image.copy()
            image_height, image_width, _ = imagednn.shape
            
            model1.setInput(cv2.dnn.blobFromImage(imagednn, size=(250, 250), swapRB=True))
            output = model1.forward()
            
            for detection in output[0, 0, :, :]:
                confidence = detection[2]
                if confidence > 0.5:
                    class_id = int(detection[1])
                    class_name = id_class_name(class_id, classNames)
                    if class_name == "person":
                        print(f"{class_id} {confidence} {class_name}")
                        box_x = int(detection[3] * image_width)
                        box_y = int(detection[4] * image_height)
                        box_width = int(detection[5] * image_width)
                        box_height = int(detection[6] * image_height)
                        box_size = box_width * box_height
                        print("box_size:", box_size)
                        
                        carState = "stop"
                        print("auto stop")
                
                        cv2.rectangle(imagednn, (box_x, box_y), (box_width, box_height), (23, 230, 210), thickness=1)
                        cv2.putText(imagednn, class_name, (box_x, box_y + int(0.05 * image_height)), cv2.FONT_HERSHEY_SIMPLEX, (0.005 * image_width), (0, 0, 255))
            
            cv2.imshow('imagednn', imagednn)
            image_ok = False
        

def main():
    global image, image_ok, carState
    
    model_path = '/home/test/Downloads/model_0517_2.h5'
    model = load_model(model_path)
    try:
        while True:
            keyValue = cv2.waitKey(1)
        
            if keyValue == ord('q'):
                break
            elif keyValue == 82:
                print("go")
                carState = "go"
            elif keyValue == 84:
                print("stop")
                carState = "stop"
            
            _, image = camera.read()
            image_ok = True
            
            preprocessed = img_preprocess(image)
            cv2.imshow('pre', preprocessed)
            X = np.asarray([preprocessed])
            steering_angle = int(model.predict(X)[0])
            print("predict angle:", steering_angle)
                
            if carState == "go":
                if 80 <= steering_angle <= 105:
                    print("go")
                    motor_go(speedSet)
                elif steering_angle > 105:
                    print("right")
                    motor_right(speedSet)
                elif steering_angle < 80:
                    print("left")
                    motor_left(speedSet)
            elif carState == "stop":
                motor_stop()
            
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    task1 = threading.Thread(target=opencvdnn_thread)
    task1.start()
    main()
    task1.join()
