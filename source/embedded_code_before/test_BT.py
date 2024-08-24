import serial

try:
    bleSerial = serial.Serial("/dev/ttyAMA10", baudrate=9600, timeout=1.0)
    bleSerial.write(b'AT')
    response = bleSerial.readline()
    if response == b'OK\r\n':
        print("Bluetooth module is working")
    else:
        print("Bluetooth module is not responding correctly")
except serial.SerialException as e:
    print(f"Serial exception: {e}")
except Exception as e:
    print(f"Error: {e}")
