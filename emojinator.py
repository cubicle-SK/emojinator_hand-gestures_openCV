import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from keras.utils import np_utils
import cv2
import os
from keras.models import load_model
from sklearn.metrics import classification_report

model= load_model('emoji.h5')
print(model)
'''
#checking whether is trained or not
data=pd.read_csv('train_ges.csv')
X=data.values[:,1:2501]/255.0
Y=data.values[:,0]
del data #to minimize memory consumption
n_class=7
x_train,x_test,y_train,y_test= train_test_split(X,Y,test_size=0.2,
                                                random_state=42)
le=LabelEncoder()
y_train=le.fit_transform(y_train)
y_test=le.transform(y_test)
y_train=np_utils.to_categorical(y_train, n_class)
y_test=np_utils.to_categorical(y_test, n_class)
x_train=x_train.reshape(x_train.shape[0],50,50,1)
x_test=x_test.reshape(x_test.shape[0],50,50,1)

print(model.summary())
print('Accuracy:', model.evaluate(x_test,y_test, verbose=1)[1])
pred=model.predict(x_test, verbose=1)

print(classification_report(np.argmax(y_test, axis=1),
                            np.argmax(pred, axis=1),
                            digits=4))
'''
def k_pred(model, image):
    process=k_pro_im(image)
    print('processed:', str(process.shape))
    pred_p= model.predict(process)[0]
    pred_class=list(pred_p).index(max(pred_p))
    pred_class=pred_class+1
    return max(pred_p), pred_class

def k_pro_im(img):
    image_x=50
    image_y=50
    img=cv2.resize(img,(image_x,image_y))
    img=np.array(img, dtype=np.float32)
    img=np.reshape(img,(-1,image_x, image_y,1))
    return img

def get_emo():
    e_folder= '/Users/saumyakansal/Desktop/SAUMYA/Python and ML/emo/'
    emoji=[]
    for emo in range(1,len(os.listdir(e_folder))):
        print('hi',emo)
        emoji.append(cv2.imread(e_folder+str(emo)+'.jpeg',-1))
        print('len:',len(emoji[emo-1]))
    return emoji

def overlay(image, emoji, x,y,w,h):
    emoji=cv2.resize(emoji,(w,h))
    try:
        print('in here')
        image[y:y+h,x:x+w]= blend(image[y:y+h , x:x+w], emoji)
        print('returning')
    except:
        print('passsing')
        pass
    return image

def blend(im, o_im):
    o_im = o_im[:,:,:3] # Grab the BRG planes
    overlay_mask = o_im[:,:,2:]  # And the alpha plane
    # Again calculate the inverse mask
    background_mask = 255- overlay_mask
    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (im * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (o_im * (1 / 255.0)) * (overlay_mask * (1 / 255.0))
    print('hello bro')

    # And finally just add them together, and rescale it back to an 8bit integer image
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))

k_pred(model, np.zeros((50, 50, 1), dtype=np.uint8))
emoji= get_emo()
cap=cv2.VideoCapture(0)
x,y,w,h= 300,50, 350, 350
while(cap.isOpened()):
    ret,imgs= cap.read()
    imgs=cv2.flip(imgs,1)
    hsv=cv2.cvtColor(imgs, cv2.COLOR_BGR2HSV)
    mask2= cv2.inRange(hsv, np.array([2,50,60]), np.array([25,150,255]))
    res= cv2.bitwise_and(imgs, imgs, mask=mask2)
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
        if cv2.contourArea(cnt)> 2500:
            x,y,w1,h1= cv2.boundingRect(cnt)
            im= thresh[y:y+h1, x:x+w1]
            im= cv2.resize(im, (50,50))
            pred_p,pred_class=k_pred(model,im)
            print(pred_class)
            print(len(emoji[pred_class-1]))
            imgs=overlay(imgs, emoji[pred_class-1], 400,250,90,90)
      
    x,y,w,h= 300,50, 350, 350    
    cv2.imshow('frame', imgs)
    cv2.imshow('contours', thresh)
    k=cv2.waitKey(10)
    if k==27:
        break
    