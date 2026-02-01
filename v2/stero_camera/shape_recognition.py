import sys 
import cv2
import numpy as np
import time

import imutils

def find_circles(frame,mask):
    contours =cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours=imutils.grab_contours(contours)
    centre =None

    if(len(contours)>0):
        c=max(contours, key=cv2.contourArea)
        ((x,y),radius)=cv2.minEnclosingCircle(c)
        M=cv2.moments(c)
        centre =(int(M["m10"]/M['m00']),int(M["m01"]/M["m00"]))

        if(radius >10):
            cv2.circle(frame,(int(x),int(y)),int(radius),(0,255,255),2)
            cv2.circle(frame,centre,5,(0,0,0),-1)

    return centre
