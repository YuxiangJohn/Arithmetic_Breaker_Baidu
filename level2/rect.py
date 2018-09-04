
# coding: utf-8



import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob




def rect(img, filename):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize = 3)
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY)
    
    median = cv2.medianBlur(binary,5)
    median = cv2.medianBlur(median,5)
    
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))
    dilation = cv2.dilate(median, element2, iterations = 1)
    erosion = cv2.erode(dilation, element1, iterations = 1)
    dilation2 = cv2.dilate(erosion, element2, iterations = 3)
    
    image1, contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    a=sorted(contours, key=cv2.contourArea, reverse=True)
    
    roilist = []
    ylist = []
    
    xlist = []
    wlist = []
    hlist = []
    
    for cnt in a[0:3]:
        x,y,w,h = cv2.boundingRect(cnt)
        #cv2.rectangle(img,(x-int(w*0.05),y-int(h*0.3)),(x+int(w*1.05),y+int(h*1.3)),(200,0,0),2)
        x-=int(w*0.05)
        if x<0:
            x=0
        y-=int(h*0.25)
        if y<0:
            y=0
        w=int(w*1.05)
        h=int(h*1.3)
        roi = img[y:y+h,x:x+w, :]
        roilist.append(roi)
        ylist.append(y)
        
        xlist.append(x)
        hlist.append(h)
        wlist.append(w)
    if ylist[1] > int(ylist[0]-hlist[0]*0.85) and ylist[1] < int(ylist[0]+hlist[0]*1.15):
        if xlist[0] <= xlist[1]:
            roilist[0] = img[ylist[0]:ylist[0]+hlist[0], xlist[0]:xlist[0]+wlist[0]+wlist[1]]
            roilist[1] = roilist[2]
            roilist.pop(2)
            
            if len(a) > 3:
                x,y,w,h = cv2.boundingRect(a[3])
                #cv2.rectangle(img,(x-int(w*0.05),y-int(h*0.3)),(x+int(w*1.05),y+int(h*1.3)),(200,0,0),2)
                x-=int(w*0.05)
                if x<0:
                    x=0
                y-=int(h*0.25)
                if y<0:
                    y=0
                w=int(w*1.05)
                h=int(h*1.3)
                roi = img[y:y+h,x:x+w, :]
                roilist.append(roi)
            
        else:
            roilist[0] = img[ylist[0]:ylist[0]+hlist[0], xlist[1]:xlist[1]+wlist[0]+wlist[1]]
            roilist[1] = roilist[2]
            roilist.pop(2)
            
            if len(a) > 3:
                x,y,w,h = cv2.boundingRect(a[3])
                #cv2.rectangle(img,(x-int(w*0.05),y-int(h*0.3)),(x+int(w*1.05),y+int(h*1.3)),(200,0,0),2)
                x-=int(w*0.05)
                if x<0:
                    x=0
                y-=int(h*0.25)
                if y<0:
                    y=0
                w=int(w*1.05)
                h=int(h*1.3)
                roi = img[y:y+h,x:x+w, :]
                roilist.append(roi)
            
    elif len(ylist)>2:  
        if ylist[2] > int(ylist[0]-hlist[0]*0.85) and ylist[2] < int(ylist[0]+hlist[0]*1.15):
            if xlist[0] <= xlist[2]:
                roilist[0] = img[ylist[0]:ylist[0]+hlist[0], xlist[0]:xlist[0]+wlist[0]+wlist[2]]
                #roilist[1] = roilist[2]
                roilist.pop(2)
            
                if len(a) > 3:
                    x,y,w,h = cv2.boundingRect(a[3])
                    #cv2.rectangle(img,(x-int(w*0.05),y-int(h*0.3)),(x+int(w*1.05),y+int(h*1.3)),(200,0,0),2)
                    x-=int(w*0.05)
                    if x<0:
                        x=0
                    y-=int(h*0.25)
                    if y<0:
                        y=0
                    w=int(w*1.05)
                    h=int(h*1.3)
                    roi = img[y:y+h,x:x+w, :]
                    roilist.append(roi)
            else:
                roilist[0] = img[ylist[0]:ylist[0]+hlist[0], xlist[2]:xlist[2]+wlist[0]+wlist[2]]
                #roilist[1] = roilist[2]
                roilist.pop(2)
            
                if len(a) > 3:
                    x,y,w,h = cv2.boundingRect(a[3])
                    #cv2.rectangle(img,(x-int(w*0.05),y-int(h*0.3)),(x+int(w*1.05),y+int(h*1.3)),(200,0,0),2)
                    x-=int(w*0.05)
                    if x<0:
                        x=0
                    y-=int(h*0.25)
                    if y<0:
                        y=0
                    w=int(w*1.05)
                    h=int(h*1.3)
                    roi = img[y:y+h,x:x+w, :]
                    roilist.append(roi)
            
            
    if len(roilist) == 2:
        cv2.imwrite("test/"+filename+"_0.png",roilist[0])
        cv2.imwrite("test/"+filename+"_1.png",roilist[1])
    elif roilist[2].shape[0] < int(roilist[1].shape[0]*0.8) or roilist[2].shape[1] < int(roilist[1].shape[0]*0.7):
        cv2.imwrite("test/"+filename+"_0.png",roilist[0])
        cv2.imwrite("test/"+filename+"_1.png",roilist[1])
    else:
        cv2.imwrite("test/"+filename+"_0.png",roilist[0])
        if ylist[1] < ylist[2]:   
            cv2.imwrite("test/"+filename+"_1.png",roilist[1])
            cv2.imwrite("test/"+filename+"_2.png",roilist[2])
        else:
            cv2.imwrite("test/"+filename+"_1.png",roilist[2])
            cv2.imwrite("test/"+filename+"_2.png",roilist[1])




def rect1(img, filename):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize = 3)
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY)
    
    median = cv2.medianBlur(binary,5)
    median = cv2.medianBlur(median,5)
    
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))
    dilation = cv2.dilate(median, element2, iterations = 1)
    erosion = cv2.erode(dilation, element1, iterations = 1)
    dilation2 = cv2.dilate(erosion, element2, iterations = 3)
    
    image1, contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    a=sorted(contours, key=cv2.contourArea, reverse=True)
    
    roilist = []
    ylist = []
    
    xlist = []
    wlist = []
    hlist = []
    
    for cnt in a[0:3]:
        x,y,w,h = cv2.boundingRect(cnt)
        #cv2.rectangle(img,(x-int(w*0.05),y-int(h*0.3)),(x+int(w*1.05),y+int(h*1.3)),(200,0,0),2)
        x-=int(w*0.05)
        if x<0:
            x=0
        y-=int(h*0.25)
        if y<0:
            y=0
        w=int(w*1.05)
        h=int(h*1.3)
        roi = img[y:y+h,x:x+w, :]
        roilist.append(roi)
        ylist.append(y)
        
        xlist.append(x)
        hlist.append(h)
        wlist.append(w)
    if ylist[1] > int(ylist[0]-hlist[0]*0.95) and ylist[1] < int(ylist[0]+hlist[0]*1.05):
        if xlist[0] <= xlist[1]:
            roilist[0] = img[ylist[0]:ylist[0]+hlist[0], xlist[0]:xlist[0]+wlist[0]+wlist[1]]
            roilist[1] = roilist[2]
            roilist.pop(2)
            
            if len(a) > 3:
                x,y,w,h = cv2.boundingRect(a[3])
                #cv2.rectangle(img,(x-int(w*0.05),y-int(h*0.3)),(x+int(w*1.05),y+int(h*1.3)),(200,0,0),2)
                x-=int(w*0.05)
                if x<0:
                    x=0
                y-=int(h*0.25)
                if y<0:
                    y=0
                w=int(w*1.05)
                h=int(h*1.3)
                roi = img[y:y+h,x:x+w, :]
                roilist.append(roi)
            
        else:
            roilist[0] = img[ylist[0]:ylist[0]+hlist[0], xlist[1]:xlist[1]+wlist[0]+wlist[1]]
            roilist[1] = roilist[2]
            roilist.pop(2)
            
            if len(a) > 3:
                x,y,w,h = cv2.boundingRect(a[3])
                #cv2.rectangle(img,(x-int(w*0.05),y-int(h*0.3)),(x+int(w*1.05),y+int(h*1.3)),(200,0,0),2)
                x-=int(w*0.05)
                if x<0:
                    x=0
                y-=int(h*0.25)
                if y<0:
                    y=0
                w=int(w*1.05)
                h=int(h*1.3)
                roi = img[y:y+h,x:x+w, :]
                roilist.append(roi)
            
    elif len(ylist)>2:  
        if ylist[2] > int(ylist[0]-hlist[0]*0.85) and ylist[2] < int(ylist[0]+hlist[0]*1.15):
            if xlist[0] <= xlist[2]:
                roilist[0] = img[ylist[0]:ylist[0]+hlist[0], xlist[0]:xlist[0]+wlist[0]+wlist[2]]
                #roilist[1] = roilist[2]
                roilist.pop(2)
            
                if len(a) > 3:
                    x,y,w,h = cv2.boundingRect(a[3])
                    #cv2.rectangle(img,(x-int(w*0.05),y-int(h*0.3)),(x+int(w*1.05),y+int(h*1.3)),(200,0,0),2)
                    x-=int(w*0.05)
                    if x<0:
                        x=0
                    y-=int(h*0.25)
                    if y<0:
                        y=0
                    w=int(w*1.05)
                    h=int(h*1.3)
                    roi = img[y:y+h,x:x+w, :]
                    roilist.append(roi)
            else:
                roilist[0] = img[ylist[0]:ylist[0]+hlist[0], xlist[2]:xlist[2]+wlist[0]+wlist[2]]
                #roilist[1] = roilist[2]
                roilist.pop(2)
            
                if len(a) > 3:
                    x,y,w,h = cv2.boundingRect(a[3])
                    #cv2.rectangle(img,(x-int(w*0.05),y-int(h*0.3)),(x+int(w*1.05),y+int(h*1.3)),(200,0,0),2)
                    x-=int(w*0.05)
                    if x<0:
                        x=0
                    y-=int(h*0.25)
                    if y<0:
                        y=0
                    w=int(w*1.05)
                    h=int(h*1.3)
                    roi = img[y:y+h,x:x+w, :]
                    roilist.append(roi)
            
            
    if len(roilist) == 2:
        cv2.imwrite("train1/"+filename+"_0.png",roilist[0])
        cv2.imwrite("train1/"+filename+"_1.png",roilist[1])
    elif roilist[2].shape[0] < int(roilist[1].shape[0]*0.8) or roilist[2].shape[1] < int(roilist[1].shape[0]*0.7):
        cv2.imwrite("train1/"+filename+"_0.png",roilist[0])
        cv2.imwrite("train1/"+filename+"_1.png",roilist[1])
    else:
        cv2.imwrite("train1/"+filename+"_0.png",roilist[0])
        if ylist[1] < ylist[2]:   
            cv2.imwrite("train1/"+filename+"_1.png",roilist[1])
            cv2.imwrite("train1/"+filename+"_2.png",roilist[2])
        else:
            cv2.imwrite("train1/"+filename+"_1.png",roilist[2])
            cv2.imwrite("train1/"+filename+"_2.png",roilist[1])




#create validate
for i in range(0,2316):
    img = cv2.imread("validate/"+str(i)+".png")
    #basename = os.path.basename(filename)
    rect(img, str(i))



# validate
for i in range(2317,4703):
    img = cv2.imread("validate/"+str(i)+".png")
    #basename = os.path.basename(filename)
    rect(img, str(i))




# validate
for i in range(4704,18919):
    img = cv2.imread("validate/"+str(i)+".png")
    #basename = os.path.basename(filename)
    rect(img, str(i))


# validate
for i in range(18920,23065):
    img = cv2.imread("validate/"+str(i)+".png")
    #basename = os.path.basename(filename)
    rect(img, str(i))


# validate
for i in range(23066,29898):
    img = cv2.imread("validate/"+str(i)+".png")
    #basename = os.path.basename(filename)
    rect(img, str(i))


# validate
for i in range(29900,29990):
    img = cv2.imread("validate/"+str(i)+".png")
    #basename = os.path.basename(filename)
    rect(img, str(i))


# validate
for i in range(29990,39968):
    img = cv2.imread("validate/"+str(i)+".png")
    #basename = os.path.basename(filename)
    rect(img, str(i))


# validate
for i in range(39969,44517):
    img = cv2.imread("validate/"+str(i)+".png")
    #basename = os.path.basename(filename)
    rect(img, str(i))


# validate
for i in range(44518,50460):
    img = cv2.imread("validate/"+str(i)+".png")
    #basename = os.path.basename(filename)
    rect(img, str(i))

# validate
for i in range(50461,54150):
    img = cv2.imread("validate/"+str(i)+".png")
    #basename = os.path.basename(filename)
    rect(img, str(i))

# validate
for i in range(54151,58756):
    img = cv2.imread("validate/"+str(i)+".png")
    #basename = os.path.basename(filename)
    rect(img, str(i))


# validate
for i in range(58757,63223):
    img = cv2.imread("validate/"+str(i)+".png")
    #basename = os.path.basename(filename)
    rect(img, str(i))


# validate
for i in range(63224,64228):
    img = cv2.imread("validate/"+str(i)+".png")
    #basename = os.path.basename(filename)
    rect(img, str(i))


# validate
for i in range(64229,70114):
    img = cv2.imread("validate/"+str(i)+".png")
    #basename = os.path.basename(filename)
    rect(img, str(i))


# validate
for i in range(70115,92216):
    img = cv2.imread("validate/"+str(i)+".png")
    #basename = os.path.basename(filename)
    rect(img, str(i))


# validate
for i in range(92217,100000):
    img = cv2.imread("validate/"+str(i)+".png")
    #basename = os.path.basename(filename)
    rect(img, str(i))

