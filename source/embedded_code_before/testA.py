import time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from gpiozero import PWMOutputDevice, DigitalOutputDevice, Motor

# 모터 핀 설정
PWMA = 12
AIN1 = 16
AIN2 = 20

PWMB = 13
BIN1 = 19
BIN2 = 26

PWMA = 12
AIN1 = 16
AIN2 = 20

PWMB = 13
BIN1 = 19
BIN2 = 26

# 모터 객체 생성 및 PWMOutputDevice 초기화
L_Motor = PWMOutputDevice(PWMA)
R_Motor = PWMOutputDevice(PWMB)

# 방향 제어 핀 설정
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

def img_preprocess(image):
    height, _, _ = image.shape
    image = image[int(height / 2):, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image = cv2.resize(image, (200, 66))
    image = cv2.GaussianBlur(image, (5, 5), 0)
    _, image = cv2.threshold(image, 160, 255, cv2.THRESH_BINARY_INV)
    image = image / 255
    return image

camera = cv2.VideoCapture(2)
camera.set(3, 640)
camera.set(4, 480)

def main():
    model_path = '/home/test/Downloads/AI_CAR/model/lane_navigation_final.h5'
    model = load_model(model_path)
    
    carState = "stop"
    speedSet = 0.95
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
            preprocessed = img_preprocess(image)
            cv2.imshow('pre', preprocessed)

            X = np.asarray([preprocessed])
            steering_angle = int(model.predict(X)[0])
            print("predict angle:", steering_angle)

            if carState == "go":
                if 70 <= steering_angle <= 95:
                    print("go")
                    motor_go(speedSet)
                elif steering_angle > 95:
                    print("right")
                    motor_right(speedSet)
                elif steering_angle < 70:
                    print("left")
                    motor_left(speedSet)
            elif carState == "stop":
                motor_stop()

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
