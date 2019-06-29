import numpy as np
import os
import cv2

im_x,im_y=50,50
cap=cv2.VideoCapture(0)
fbag= cv2.createBackgroundSubtractorMOG2()
def f(f_n):
    if not os.path.exists(f_n):
        os.mkdir(f_n)

def store_im(g_id):
    total=1200
    cap=cv2.VideoCapture(0)
    x,y,w,h= 300,50,350,350
  
    f('ges/'+str(g_id))
    pic=0
    flag_start_capturing=False
    frames=0
  
    while True:
        ret,frame= cap.read()
        frame=cv2.flip(frame,1)
        hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask2= cv2.inRange(hsv, np.array([2,50,60]), np.array([25,150,255]))
        res= cv2.bitwise_and(frame, frame, mask=mask2)
        gray= cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        median= cv2.GaussianBlur(gray,(5,5),0)
    
        kernel_square= np.ones((5,5), np.uint8)
        dilation= cv2.dilate(median, kernel_square, iterations=2)
        opening= cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel_square)
    
        ret, thresh= cv2.threshold(opening, 30, 255,cv2.THRESH_BINARY)
        thresh=thresh[y:y+h , x:x+w]
        contours= cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
           
        if len(contours)>0:
            cnt= max(contours, key= cv2.contourArea)
            print('fr:',frames)
            if cv2.contourArea(cnt)> 10000 and frames >50:
                x1,y1,w1,h1= cv2.boundingRect(cnt)
                pic+=1
                save= thresh[y1:y1+ h1,x1:x1+ w1]
                save= cv2.resize(save, (im_x, im_y))
                cv2.putText(frame,'capturing....', (30,60), cv2.FONT_HERSHEY_TRIPLEX,2,(127,255,255))
                cv2.imwrite('ges/'+str(g_id)+'/'+str(pic)+'.jpg',save)
             
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)
        cv2.putText(frame, str(pic), (30,400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127,127,255))
        cv2.imshow('capturing gesture', frame)
        cv2.imshow('thresh', thresh)
        keypress= cv2.waitKey(1)
        if keypress== ord('c'):
            if flag_start_capturing== False:
                flag_start_capturing=True
            else:
                flag_start_capturing=False
                frames=0
        if flag_start_capturing==True:
            frames+=1
        if pic==total:
            break
#os.mkdir('ges')
g_id= input('enter no:')
store_im(g_id)           
        