import cv2
import numpy as np
import math

def nothing(x):
    pass

def rescale_frame(frame, percent= 100):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

width = 1280
height = 720
w1=0
w2=100
h1=1100
h2=1280
clc = cv2.imread('Clear-1.jpg')
clc = cv2.resize(clc,(h2-h1,w2-w1), interpolation =cv2.INTER_AREA)
cap = cv2.VideoCapture(1)
change_res(width,height)
cv2.namedWindow('opened')
cv2.createTrackbar('black', 'opened', 170, 255, nothing)
cv2.createTrackbar('area_max', 'opened', 600, 5000, nothing)
cv2.createTrackbar('area_min', 'opened', 10, 500, nothing)
cv2.createTrackbar('radii', 'opened',2000,3000, nothing)
kernel = np.ones((5, 5), np.uint8)
center = []
ret, frame = cap.read()
rows, cols, ch = frame.shape
screen = np.zeros((rows,cols, ch), np.uint8)
screen[w1:w2, h1:h2] = clc
c = 0
while (1):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    black = cv2.getTrackbarPos('black', 'opened')
    _, thresh = cv2.threshold(gray,black,255,cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(thresh, (5, 5), 0)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    img, contours, heirarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = []
    for cnt in contours:
        area.append(cv2.contourArea(cnt))
        ar = cv2.contourArea(cnt)
        min_val = cv2.getTrackbarPos('area_min', 'opened')
        max_val = cv2.getTrackbarPos('area_max', 'opened')

        if (ar < max_val and ar > min_val):
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if(len(center) > 0):
                rad = cv2.getTrackbarPos('radii', 'opened')
                temp1, temp2 = center.pop()
                dis = ((x - temp1) * (x - temp1) + (y - temp2) * (y - temp2))
                center.append([temp1,temp2])
                dis = math.sqrt(dis)
                print dis
                if(dis < rad):
                    center.append([int(x), int(y)])
            else:
                center.append([int(x), int(y)])
            if (y<w2 and y>w1 and x<h2 and x>h1):
                c = 32
            if (len(center) > 1):
                x1, y1 = center.pop()
                x2, y2 = center.pop()
                screen = cv2.line(screen, (x1, y1), (x2, y2), (0, 255, 255), 5)
                center.append([x1, y1])
        else:
            while(len(center)!=0):
                center.pop()

    if(len(area)==0):
        while (len(center) != 0):
            center.pop()

    print area

    cv2.imshow('opened', opening)
    cv2.imshow('screen', screen)


    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    elif k == 32 or c==32:
        c = 0
        screen = np.zeros((rows, cols, ch), np.uint8)
        screen[h1:h2, w1:w2] = clc
        if (len(center) > 0):
            center.pop()

cv2.destroyAllWindows()