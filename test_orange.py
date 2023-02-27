import cv2
import numpy as np

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_FOCUS, 70)

isMask = True

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # lower_orange = (0, 100, 100)
    # upper_orange = (20, 255, 255)
    ORANGE_MIN = np.array([0, 80, 192], np.uint8)
    ORANGE_MAX = np.array([5, 255, 255], np.uint8)
    ORANGE_MIN2 = np.array([174, 80, 192], np.uint8)
    ORANGE_MAX2 = np.array([179, 255, 255], np.uint8)
    if isMask:
        mask = cv2.inRange(hsv, ORANGE_MIN, ORANGE_MAX)
        mask2 = cv2.inRange(hsv, ORANGE_MIN2, ORANGE_MAX2)

        mask = cv2.bitwise_or(mask, mask2)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=6)
        # mask = cv2.GaussianBlur(mask, (3, 3), 0)
        # mask = cv2.bitwise_and(frame, frame, mask=mask)



    else:
        mask = frame

    cv2.imshow('YOLOv8 Detection', mask)
    pressed = cv2.waitKey(5) & 0xFF
    if pressed == 27:
        break
    elif pressed == ord('c'):
        isMask = not isMask
cap.release()
cv2.destroyAllWindows()