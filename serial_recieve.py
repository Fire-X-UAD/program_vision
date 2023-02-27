import serial

ser = serial.Serial('COM10', 9600, timeout=0, parity=serial.PARITY_NONE, rtscts=1)
print(ser.read(100))