from gpiozero import PWMOutputDevice, DigitalOutputDevice
from time import sleep

PWMA = 12
AIN1 = 16
AIN2 = 20

PWMB = 13
BIN1 = 19
BIN2 = 26

# 모터 객체 생성
L_Motor = PWMOutputDevice(PWMA)
R_Motor = PWMOutputDevice(PWMB)

# 방향 제어 핀 설정
AIN1_pin = DigitalOutputDevice(AIN1)
AIN2_pin = DigitalOutputDevice(AIN2)
BIN1_pin = DigitalOutputDevice(BIN1)
BIN2_pin = DigitalOutputDevice(BIN2)

try:
    while True:
        # 전진
        print("전진")
        AIN1_pin.off()
        AIN2_pin.on()
        L_Motor.value = 1.0
        BIN1_pin.off()
        BIN2_pin.on()
        R_Motor.value = 1.0
        sleep(1.0)
        
        # 정지
        print("정지")
        L_Motor.value = 0
        R_Motor.value = 0
        sleep(2.0)
        
except KeyboardInterrupt:
    pass

# GPIO 리소스 정리
L_Motor.close()
R_Motor.close()
AIN1_pin.close()
AIN2_pin.close()
BIN1_pin.close()
BIN2_pin.close()

