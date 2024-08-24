import threading
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from gpiozero import PWMOutputDevice, DigitalOutputDevice
import os
import time
from queue import Queue

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

speedSet = 0.6
classNames = {0: 'normal', 1: 'burned'}

camera = cv2.VideoCapture(0)
camera2 = cv2.VideoCapture(2)
camera.set(3, 640)
camera.set(4, 480)
camera2.set(3, 640)
camera2.set(4, 480)

image_queue = Queue(maxsize=1)
image2_queue = Queue(maxsize=1)

image_ok = False
carState = "stop"
running = True

def motor_control(command, speed=0.7):
    if command == "go":
        L_Motor.value = speed
        AIN2_pin.on()
        AIN1_pin.off()
        R_Motor.value = speed
        BIN2_pin.on()
        BIN1_pin.off()
    elif command == "back":
        L_Motor.value = speed
        AIN2_pin.off()
        AIN1_pin.on()
        R_Motor.value = speed
        BIN2_pin.off()
        BIN1_pin.on()
    elif command == "stop":
        L_Motor.value = 0
        R_Motor.value = 0
    elif command == "right":
        L_Motor.value = speed
        AIN2_pin.on()
        AIN1_pin.off()
        R_Motor.value = speed
        BIN2_pin.off()
        BIN1_pin.on()
    elif command == "left":
        L_Motor.value = speed
        AIN2_pin.off()
        AIN1_pin.on()
        R_Motor.value = speed
        BIN2_pin.on()
        BIN1_pin.off()

def img_preprocess(image):
    height, width, _ = image.shape
    image = image[height // 2:, :, :]  # Use integer division
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image = cv2.resize(image, (200, 66))
    image = cv2.GaussianBlur(image, (5, 5), 0)
    _, image = cv2.threshold(image, 160, 255, cv2.THRESH_BINARY_INV)
    image = image.astype(np.float32) / 255.0  # Use float32 for better performance
    return image

def capture_and_preprocess():
    global running
    while running:
        ret1, img1 = camera.read()
        ret2, img2 = camera2.read()
        if ret1:
            img1 = img1[:, 100:540, :]
            if not image_queue.full():
                image_queue.put(img1)
        if ret2:
            if not image2_queue.full():
                image2_queue.put(img2)

def opencvdnn_thread():
    global running, carState
    file_path = '/home/test/test/test_auto_stop/'
    i = 0
    model = YOLO("/home/test/model-test/models/ciga_changseop_ncnn_model")
    while running:
        if not image_queue.empty():
            imagednn = image_queue.get()
            results = model(imagednn)
            annotated_frame = imagednn.copy()

            for result in results:
                try:
                    annotated_frame = result.plot()
                    if result.boxes.conf[result.boxes.conf > 0.7].any() and carState == 'go':
                        print('auto stop')
                        cv2.imwrite("%s_%05d.png" % (file_path, i), annotated_frame)
                        i += 1
                        carState = 'stop'  # Adding stop to avoid continuous saving
                except Exception as e:
                    print(f"Error plotting results: {e}")

def main():
    global carState, running
    model_path = '/home/test/Downloads/AI_CAR/model/lane_navigation_final.h5'
    model = load_model(model_path)
    try:
        while running:
            if not image2_queue.empty():
                image2 = image2_queue.get()
                preprocessed = img_preprocess(image2)
                X = np.asarray([preprocessed])
                try:
                    steering_angle = int(model.predict(X)[0])
                    print("predict angle:", steering_angle)
                except Exception as e:
                    print(f"Prediction error: {e}")
                    continue

                if carState == "go":
                    if 80 <= steering_angle <= 105:
                        print("go")
                        motor_control("go", speedSet)
                    elif steering_angle > 105:
                        print("right")
                        motor_control("right", speedSet)
                    elif steering_angle < 80:
                        print("left")
                        motor_control("left", speedSet)
                elif carState == "stop":
                    motor_control("stop")

                cv2.imshow('preprocessed', preprocessed)
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
