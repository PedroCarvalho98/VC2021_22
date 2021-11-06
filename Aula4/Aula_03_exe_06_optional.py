# Aula_03_ex_06_optional.py

import sys
import numpy as np
import cv2

capture = cv2.VideoCapture(0)
while(True):
    ret, frame = capture.read()
    cv2.imshow('video', frame)

    # Canny Operator 3 x 3 - X
    edges3x3_X = cv2.Canny(frame, 60, 100, 3)
    # Canny Operator 3 x 3 - Y
    edges3x3_Y = cv2.Canny(frame, 60, 100, 3)
    # Canny Operator 3x3 - Result
    result3x3 = (edges3x3_X ** 2 + edges3x3_Y ** 2) ** 0.5
    cv2.imshow("Canny 3 x 3 - Result", result3x3)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
