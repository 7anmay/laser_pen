import cv2
import numpy as np
def nothing(x): pass
cap = cv2.VideoCapture(1)
zero = 0
#f = open("thresh_val.txt","w+")
#####
f = open("thresh_val.txt","r+")
#f.seek(0)
val = f.read().split()
print(val)
print(len(val))
if(len(val)!= 3):
    for i in range(3):
        f.write("%d\r" % zero)
    f.seek(0)
    val = f.readlines()
#print(val)
print(int(val[0]))
#####

cv2.namedWindow('opened')
cv2.createTrackbar('black','opened',int(val[0]),255,nothing)
cv2.createTrackbar('area_max','opened',int(val[1]),5000,nothing)
cv2.createTrackbar('area_min','opened',int(val[2]),500,nothing)
kernel = np.ones((5,5),np.uint8)
center = []
ret, frame = cap.read()
rows,cols,ch = frame.shape
screen = np.zeros((rows,cols,ch), np.uint8)
t = 0
while(1):
    if t==1:
        for i in range(10):
            ret, frame = cap.read()
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    black = cv2.getTrackbarPos('black','opened')
    _, thresh = cv2.threshold(gray,black,255,cv2.THRESH_BINARY)

    blur = cv2.GaussianBlur(thresh,(5,5),0)
    opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
    contours, heirarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = []
    for cnt in contours:
        t = 0
        area.append(cv2.contourArea(cnt))
        ar = cv2.contourArea(cnt)
        min_val = cv2.getTrackbarPos('area_min', 'opened')
        max_val = cv2.getTrackbarPos('area_max', 'opened')

        if (ar < max_val and ar > min_val):
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center.append([int(x),int(y)])

            if (len(center) > 1):
                x1,y1 = center.pop()
                x2,y2 = center.pop()
                screen = cv2.line(screen,(x1,y1),(x2,y2),(0,255,255),5)
                center.append([x1,y1])
    print (area)

    cv2.imshow('opened', opening)
    cv2.imshow('screen',screen)

    t=0

    k = cv2.waitKey(1) & 0xFF
    if k==27:
        break
    elif k==32:
        screen = np.zeros((rows, cols, ch), np.uint8)
        if(len(center)>0):
            center.clear()
    elif (k == 116 or k == 84):
        f.seek(0)
        f.write("%d\r" % black)
        f.write("%d\r" % max_val)
        f.write("%d\r" % min_val)

    elif (k== 115 or k== 83):
        cv2.waitKey(0)
        if (len(center)>0):
            center.pop()
        t=1
f.close()
cv2.destroyAllWindows()