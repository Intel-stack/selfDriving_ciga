import cv2
from gpiozero import PWMOutputDevice, DigitalOutputDevice
from time import sleep

# 모터 핀 설정
PWMA = 13
AIN1 = 19
AIN2 = 26

PWMB = 12
BIN1 = 20
BIN2 = 16

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

def main():
    camera = cv2.VideoCapture(0)
    camera.set(3, 640)
    camera.set(4, 480)
    filepath_go = "/home/test/test/tempImage/goImage/"
    filepath_left = "/home/test/test/tempImage/leftImage/"
    filepath_right = "/home/test/test/tempImage/rightImage/"
    i = 0
    carState = "stop"
    #motor_go(1.0)
    speedSet = 1.0  # 모터 속도 설정
    
    while camera.isOpened():
        keyValue = cv2.waitKey(10)
        
        if keyValue == ord('q'):
            break
        elif keyValue == 82:
            print("go")
            carState = "go"
            motor_go(speedSet)
        elif keyValue == 84:
            print("stop")
            carState = "stop"
            motor_stop()
        elif keyValue == 81:
            print("left")
            carState = "left"
            motor_left(speedSet)
        elif keyValue == 83:
            print("right")
            carState = "right"
            motor_right(speedSet)

        _, image = camera.read()
        #cv2.rectangle(image,(0,image.shape[0]//2-10), (image.shape[1], image.shape[0]//2+10), (0,0,0), 3, cv2.LINE_AA)
        #cv2.rectangle(image,(image.shape[1]//2-10,0), (image.shape[1]//2+10,image.shape[0]), (0,0,0), 3, cv2.LINE_AA)
        cv2.imshow('Original', image)
        
        height, _, _ = image.shape
        save_image = image[:,:,:]
        #save_image = cv2.cvtColor(save_image, cv2.COLOR_BGR2YUV)
        save_image = cv2.GaussianBlur(save_image, (5, 5), 0)
        save_image = cv2.resize(save_image, (128,96))
        _,save_image = cv2.threshold(save_image,95,255,cv2.THRESH_BINARY)
        cv2.imshow('Save', save_image)
        
        if carState == "left":
            cv2.imwrite("%s_%05d_%03d.png" % (filepath_left, i, 45), save_image)
            i += 1
        elif carState == "right":
            cv2.imwrite("%s_%05d_%03d.png" % (filepath_right, i, 135), save_image)
            i += 1
        elif carState == "go":
            cv2.imwrite("%s_%05d_%03d.png" % (filepath_go, i, 90), save_image)
            i += 1
        
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
