import gpiod
import time

PWMA = 18
AIN1 = 22
AIN2 = 27

PWMB = 23
BIN1 = 25
BIN2 = 24

# chipname = "gpiochip0"
chipname = "gpiochip4"

# GPIO chip 열기
chip = gpiod.Chip(chipname)

# GPIO 핸들 얻기
line_AIN1 = chip.get_line(AIN1)
line_AIN2 = chip.get_line(AIN2)
line_PWMA = chip.get_line(PWMA)

line_BIN1 = chip.get_line(BIN1)
line_BIN2 = chip.get_line(BIN2)
line_PWMB = chip.get_line(PWMB)

# line 설정
line_AIN1.request(consumer="script", type=gpiod.LINE_REQ_DIR_OUT)
line_AIN2.request(consumer="script", type=gpiod.LINE_REQ_DIR_OUT)
line_PWMA.request(consumer="script", type=gpiod.LINE_REQ_PWM)

line_BIN1.request(consumer="script", type=gpiod.LINE_REQ_DIR_OUT)
line_BIN2.request(consumer="script", type=gpiod.LINE_REQ_DIR_OUT)
line_PWMB.request(consumer="script", type=gpiod.LINE_REQ_PWM)

# PWM frequency 설정
line_PWMA.set_config(gpiod.LINE_FLAG_OPEN_DRAIN, 0, 500000)  # 500Hz PWM 주파수
line_PWMB.set_config(gpiod.LINE_FLAG_OPEN_DRAIN, 0, 500000)  # 500Hz PWM 주파수

# PWM duty cycle 설정
line_PWMA.request(consumer="script", type=gpiod.LINE_REQ_PWM)
line_PWMB.request(consumer="script", type=gpiod.LINE_REQ_PWM)

# PWM duty cycle 설정 함수
def set_pwm(line, duty_cycle):
    line.request(consumer="script", type=gpiod.LINE_REQ_PWM)
    line.set_config(gpiod.LINE_FLAG_OPEN_DRAIN, duty_cycle, 500000)

try:
    while True:
        # 전진
        line_AIN1.set_value(0)
        line_AIN2.set_value(1)
        set_pwm(line_PWMA, 100)
        line_BIN1.set_value(0)
        line_BIN2.set_value(1)
        set_pwm(line_PWMB, 100)
        time.sleep(1.0)
        
        # 정지
        set_pwm(line_PWMA, 0)
        set_pwm(line_PWMB, 0)
        time.sleep(1.0)
        
except KeyboardInterrupt:
    pass

# GPIO 리소스 정리
line_AIN1.release()
line_AIN2.release()
line_PWMA.release()

line_BIN1.release()
line_BIN2.release()
line_PWMB.release()

chip.close()

