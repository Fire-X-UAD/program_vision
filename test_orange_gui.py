import cv2
import numpy as np

def nothing(x):
    pass


cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_FOCUS, 70)


enad_val = True

# Create a window
cv2.namedWindow('image')
ORANGE_MIN = np.array([0, 92, 192], np.uint8)
ORANGE_MAX = np.array([5, 255, 255], np.uint8)
# Create trackbars for color change
# Hue is from 0-179 for Opencv
cv2.createTrackbar('HMin', 'image', ORANGE_MIN[0], 179, nothing)
cv2.createTrackbar('SMin', 'image', ORANGE_MIN[1], 255, nothing)
cv2.createTrackbar('VMin', 'image', ORANGE_MIN[2], 255, nothing)
cv2.createTrackbar('HMax', 'image', ORANGE_MAX[0], 179, nothing)
cv2.createTrackbar('SMax', 'image', ORANGE_MAX[1], 255, nothing)
cv2.createTrackbar('VMax', 'image', ORANGE_MAX[2], 255, nothing)


# Initialize HSV min/max values
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

while(1):
    ret, frame = cap.read()
    image = frame
    # Get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin', 'image')
    sMin = cv2.getTrackbarPos('SMin', 'image')
    vMin = cv2.getTrackbarPos('VMin', 'image')
    hMax = cv2.getTrackbarPos('HMax', 'image')
    sMax = cv2.getTrackbarPos('SMax', 'image')
    vMax = cv2.getTrackbarPos('VMax', 'image')

    # Set minimum and maximum HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Convert to HSV format and color threshold
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    if enad_val:
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=1)
    result = cv2.bitwise_and(image, image, mask=mask)

    # Print if there is a change in HSV value
    if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
        print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax

    # Display result image
    cv2.imshow('image', result)
    pressed = cv2.waitKey(10) & 0xFF
    if pressed == ord('q'):
        break
    elif pressed == ord('c'):
        enad_val = not enad_val

cv2.destroyAllWindows()