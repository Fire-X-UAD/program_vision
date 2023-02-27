import serial

ser = serial.Serial('COM10', 9600, timeout=0, parity=serial.PARITY_NONE, rtscts=1)
while 1:
    ser.write(b"hellooooo\n")