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
                    print "object position: " + format(obj.getPositionAtInstant(frameNum).project(homography))
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
    drawTrajectory(frame, fNo)
    for i in range(len(cArray)):
        cv2.circle(frame, (cArray[i][0], cArray[i][1]), 3, (0, 255, 0), -1)
    cv2.imshow('Video', frame)
    fNo += 1
    if cv2.waitKey(25) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
