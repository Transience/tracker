__author__ = 'Akhil'


import cv2
import storage
from numpy.linalg.linalg import inv
from numpy import loadtxt

homographyFilename = "laurier-homography.txt"
homography = inv(loadtxt(homographyFilename))
databaseFilename = "laurier.sqlite"
trajectoryType = "object"
objects = storage.loadTrajectoriesFromSqlite(databaseFilename, trajectoryType)
features = storage.loadTrajectoriesFromSqlite(databaseFilename, "feature")

drawing = False   # true if mouse is pressed
cArray = []
fNo = 0   # stores the frame number

def findObject(frameNum, x=0, y=0):
    box = []
    global width, height, objects, features
    for obj in objects:
        if obj.existsAtInstant(frameNum):
            objFeatures = [features[i] for i in obj.featureNumbers]
            u = []
            v = []
            for f in objFeatures:
                if f.existsAtInstant(frameNum):
                    projectedPosition = f.getPositionAtInstant(frameNum).project(homography)
                    u.append(projectedPosition.x)
                    v.append(projectedPosition.y)
            xmin = min(u)
            xmax = max(u)
            ymin = min(v)
            ymax = max(v)
            if x == 0 and y == 0:
                box.append([ymax, ymin, xmax, xmin])
            if xmax > x > xmin and ymax > y > ymin:
                    print "object detected: " + format(obj.getNum())
    return box

def drawBox(frame, frameNum):
    box = findObject(frameNum)
    for i in range(len(box)):
        cv2.line(frame, (box[i][3], box[i][0]), (box[i][2], box[i][0]), (255, 0, 0), 3)
        cv2.line(frame, (box[i][3], box[i][1]), (box[i][2], box[i][1]), (255, 0, 0), 3)
        cv2.line(frame, (box[i][3], box[i][1]), (box[i][3], box[i][0]), (255, 0, 0), 3)
        cv2.line(frame, (box[i][2], box[i][1]), (box[i][2], box[i][0]), (255, 0, 0), 3)

def coordinates(event,x,y,flags,param):
    global drawing, cArray, fNo
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        print x, y
        cArray.append([x, y])
        findObject(fNo, x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            print x, y
            cArray.append([x, y])
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        del cArray[:]

cap = cv2.VideoCapture('laurier.avi')
cv2.namedWindow('Video')
cv2.setMouseCallback('Video', coordinates)
width = cap.get(3)
height = cap.get(4)

while(cap.isOpened()):
    ret, frame = cap.read()
    drawBox(frame, fNo)
    for i in range(len(cArray)):
        cv2.circle(frame, (cArray[i][0], cArray[i][1]), 3, (0, 255, 0), -1)
    cv2.imshow('Video', frame)
    fNo += 1
    if cv2.waitKey(25) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
