import cv2                
import time
import sys, os

sys.path.append(os.path.join(sys.path[0],'emotion_detection'))
sys.path.append(os.path.join(sys.path[0],'evm'))
from emotion_detection.emotions import emotion_detection
from evm.magnification import Magnify

fps = 8.
alpha = 300
lambda_c = 200
fl = 0.3
fh = 1.
cam = cv2.VideoCapture(0)
_,img = cam.read()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
s = Magnify(gray,alpha,lambda_c,fl,fh,fps)


while True:
    t1 = time.process_time()
    _,img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    evm = s.Magnify(gray)

    cv2.imshow('output', evm)
    # cv2.imshow('outpet', p1)
    # cv2.imshow('gray2', grayz)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    t2 = time.process_time()
    
    print("set fps",1/(t2-t1))

cam.release()
cv2.destroyAllWindows()