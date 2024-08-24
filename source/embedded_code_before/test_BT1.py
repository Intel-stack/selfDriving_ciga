import serial
#시리얼을 불러온다.

BlSerial = serial.Serial("/dev/ttyAMA10", baudrate=9600, timeout=1.0)
#dev/ttyS0을 시리얼통신핀으로 사용한다. 9600의 통신속도를 가지고, 1초이상 데이터가 들어오지않으면 timeout된다.

try:
    while True:
        sendData = "nice to meet you! \r\n"   #nice to meet you!라는 메세지. 즉 데이터를 보낸다.
        BlSerial.write(sendData.encode())  #write를 이용해서 위의 메세지를 시리얼 데이터로 전송한다.
        data = BlSerial.readline() #한줄씩 값을 받는다. 
        data = data.decode()  #decode로 시리얼통신의 bytes 타입을 문자열 타입으로 변경한다.
        print(data)  #data를 출력한다.
        
except KeyboardInterrupt:          #키보드에서 Ctrl+c가 눌려지면 종료합니다.
    pass

BlSerial.close()
