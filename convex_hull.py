import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray,(35,35),0)

    _, thresh = cv2.threshold(blur,60,255,cv2.THRESH_BINARY_INV)

    contours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:

        cnt = max(contours, key=cv2.contourArea)

        hull = cv2.convexHull(cnt)

        cv2.drawContours(frame,[cnt],-1,(0,255,0),2)
        cv2.drawContours(frame,[hull],-1,(255,0,0),2)

    cv2.imshow("frame",frame)

    if cv2.waitKey(1)==27:
        break