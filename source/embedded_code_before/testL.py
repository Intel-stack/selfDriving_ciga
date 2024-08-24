import cv2
import numpy as np
from gpiozero import PWMOutputDevice, DigitalOutputDevice

PWMA = 12
AIN1 = 16
AIN2 = 20

PWMB = 13
BIN1 = 19
BIN2 = 26

# Motor setup using gpiozero
L_Motor = PWMOutputDevice(PWMA)
R_Motor = PWMOutputDevice(PWMB)

AIN1_device = DigitalOutputDevice(AIN1)
AIN2_device = DigitalOutputDevice(AIN2)
BIN1_device = DigitalOutputDevice(BIN1)
BIN2_device = DigitalOutputDevice(BIN2)

def motor_go(speed):
    L_Motor.value = speed / 100.0
    AIN2_device.on()
    AIN1_device.off()
    R_Motor.value = speed / 100.0
    BIN2_device.on()
    BIN1_device.off()
    
def motor_right(speed):
    L_Motor.value = speed / 100.0
    AIN2_device.on()
    AIN1_device.off()
    R_Motor.value = 0
    BIN2_device.off()
    BIN1_device.on()
    
def motor_left(speed):
    L_Motor.value = 0
    AIN2_device.off()
    AIN1_device.on()
    R_Motor.value = speed / 100.0
    BIN2_device.on()
    BIN1_device.off()

def main():
    camera = cv2.VideoCapture(0)
    camera.set(3, 160) 
    camera.set(4, 120)

    while camera.isOpened():
        ret, frame = camera.read()
        #frame = cv2.flip(frame, -1)
        cv2.imshow('normal', frame)
        
        crop_img = frame[60:120, 0:160]
        
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        ret, thresh1 = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)
        
        mask = cv2.erode(thresh1, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cv2.imshow('mask', mask)
    
        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            if cx >= 95 and cx <= 125:              
                print("Turn Left!")
                motor_left(100)
            elif cx >= 39 and cx <= 65:
                print("Turn Right")
                motor_right(100)
            else:
                print("Go")
                motor_go(100)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
