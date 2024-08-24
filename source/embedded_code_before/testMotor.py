from gpiozero import PWMOutputDevice, DigitalOutputDevice
from time import sleep

PWMA = 18
AIN1 = 22
AIN2 = 27

PWMB = 23
BIN1 = 25
BIN2 = 24

# 모터 설정
L_Motor = PWMOutputDevice(PWMA)
AIN1_Motor = DigitalOutputDevice(AIN1)
AIN2_Motor = DigitalOutputDevice(AIN2)

R_Motor = PWMOutputDevice(PWMB)
BIN1_Motor = DigitalOutputDevice(BIN1)
BIN2_Motor = DigitalOutputDevice(BIN2)

# 모터 정지 함수
def stop_motors():
    L_Motor.value = 0
    R_Motor.value = 0

try:
    while True:
        # 전진
        AIN1_Motor.off()
        AIN2_Motor.on()
        L_Motor.value = 0.5
        
        BIN1_Motor.off()
        BIN2_Motor.on()
        R_Motor.value = 0.5
        sleep(1)
        
        # 정지
        stop_motors()
        sleep(1)
        
except KeyboardInterrupt:
    pass

# GPIO 정리
L_Motor.close()
R_Motor.close()
AIN1_Motor.close()
AIN2_Motor.close()
BIN1_Motor.close()
BIN2_Motor.close()
