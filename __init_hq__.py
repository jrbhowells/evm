import cv2                
import time
import sys, os
import numpy as np

sys.path.append(os.path.join(sys.path[0],'emotion_detection'))
sys.path.append(os.path.join(sys.path[0],'evm'))
from emotion_detection.emotions import emotion_detection
from evm.magnification import Magnify

fps = 8.
alpha = 300
lambda_c = 200
fl = 0.3
fh = 1.
cam = cv2.VideoCapture('woman2.mp4')
_,img = cam.read()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
s = Magnify(gray,alpha,lambda_c,fl,fh,fps)

# Output files for videos
evm_out = cv2.VideoWriter('evm_out.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30.0, (1280,720))
emo_out = cv2.VideoWriter('emo_out.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30.0, (1280,720))

# Desired number of frames to export
length = 1000

# Helpers: current frame and time
fr = 0
t1 = time.process_time()

while True:
    try: # Catches and saves if we run out of frames - detecting no of frames is hard
        _,img = cam.read()

        evmb = s.Magnify(img[:,:,0]) # Blue component, magnified
        evmg = s.Magnify(img[:,:,1]) # Green component, magnified
        evmr = s.Magnify(img[:,:,2]) # Red component, magnified

        # Combine magnified elements to full colour
        bgr = np.dstack((evmb, evmg, evmr))

        raw_emo = emotion_detection(img)
        evm_emo = emotion_detection(bgr)

        # Write frames to files
        evm_out.write(evm_emo)
        emo_out.write(raw_emo)

    except:
        print("Ran out of frames.")
        break

    fr += 1
    print("frame", fr, "/", length)
    if (fr >= length):
        break

print("Exported", fr, "frames in", time.process_time() - t1, "seconds")
cam.release()
evm_out.release()
emo_out.release()