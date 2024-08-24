import gpiod
import time

SW1 = 5
chipname = "gpiochip0"

# GPIO chip 열기
chip = gpiod.Chip(chipname)

# GPIO 핸들 얻기
line_SW1 = chip.get_line(SW1)

# line 설정
line_SW1.request(consumer="script", type=gpiod.LINE_REQ_DIR_IN)

try:
    while True:
        sw1Value = line_SW1.get_value()
        print(sw1Value)
        time.sleep(0.1)

except KeyboardInterrupt:
    pass

# GPIO 리소스 정리
line_SW1.release()
chip.close()

