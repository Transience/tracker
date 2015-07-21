__author__ = 'Akhil'

import cv2
import storage
from numpy.linalg.linalg import inv
from numpy import loadtxt

homographyFilename = "laurier-homography.txt"
homography = inv(loadtxt(homographyFilename))
databaseFilename = "laurier.sqlite"
objects = storage.loadTrajectoriesFromSqlite(databaseFilename, "object")
features = storage.loadTrajectoriesFromSqlite(databaseFilename, "feature")

drawing = False   # true if mouse is pressed
cArray = []   # stores new trajectory positions (temporary)
fNo = 0   # stores the frame number
tArray = []   # stores old trajectory positions
objTag = None   # stores selected object's id
edit = False   # turns on edit mode if true

def findObject(frameNum, x=None, y=None):
    box = []
    global objects, features, objTag
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
            if x is None and y is None:
                box.append([ymax, ymin, xmax, xmin, obj.getNum()])
            if xmax > x > xmin and ymax > y > ymin:
                print "object detected: " + format(obj.getNum())
                print "object position: " + format(obj.getPositionAtInstant(frameNum).project(homography))
                objTag = obj.getNum()
    return box

def drawTrajectory(frame, frameNum):
    global objects, tArray
    for obj in objects:
        if obj.existsAtInstant(frameNum):
            prevPosition = obj.getPositionAtInstant(obj.getFirstInstant()).project(homography)
            for pos in obj.getPositions():
                position = pos.project(homography)
                cv2.line(frame, (position[0], position[1]), (prevPosition[0], prevPosition[1]), (0, 0, 255), 2)
                prevPosition = position
                if pos[0] == obj.getPositionAtInstant(frameNum)[0] and pos[1] == obj.getPositionAtInstant(frameNum)[1]:
                    break

def drawBox(frame, frameNum):
    global objTag
    box = findObject(frameNum)
    for i in range(len(box)):
        if box[i][4] == objTag:
            cv2.line(frame, (box[i][3], box[i][0]), (box[i][2], box[i][0]), (0, 255, 255), 3)
            cv2.line(frame, (box[i][3], box[i][1]), (box[i][2], box[i][1]), (0, 255, 255), 3)
            cv2.line(frame, (box[i][3], box[i][1]), (box[i][3], box[i][0]), (0, 255, 255), 3)
            cv2.line(frame, (box[i][2], box[i][1]), (box[i][2], box[i][0]), (0, 255, 255), 3)
        else:
            cv2.line(frame, (box[i][3], box[i][0]), (box[i][2], box[i][0]), (255, 0, 0), 3)
            cv2.line(frame, (box[i][3], box[i][1]), (box[i][2], box[i][1]), (255, 0, 0), 3)
            cv2.line(frame, (box[i][3], box[i][1]), (box[i][3], box[i][0]), (255, 0, 0), 3)
            cv2.line(frame, (box[i][2], box[i][1]), (box[i][2], box[i][0]), (255, 0, 0), 3)

def drawEditBox(frame):
    global width, height, edit
    if edit == True:
        cv2.line(frame, (0, 0), (width, 0), (0, 255, 255), 3)
        cv2.line(frame, (0, 0), (0, height), (0, 255, 255), 3)
        cv2.line(frame, (width, height), (width, 0), (0, 255, 255), 3)
        cv2.line(frame, (width, height), (0, height), (0, 255, 255), 3)

def coordinates(event, x, y, flags, param):
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
width = int(cap.get(3))
height = int(cap.get(4))

while(cap.isOpened()):
    ret, frame = cap.read()
    drawBox(frame, fNo)
    drawTrajectory(frame, fNo)
    drawEditBox(frame)
    k = cv2.waitKey(33) & 0xFF
    if k == 27:
        break
    if k == 101:
        edit = edit != True
    for i in range(len(cArray)):
        cv2.circle(frame, (cArray[i][0], cArray[i][1]), 3, (0, 255, 0), -1)
    cv2.imshow('Video', frame)
    fNo += 1

cap.release()
cv2.destroyAllWindows()
