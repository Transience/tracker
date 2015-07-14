__author__ = 'Akhil'


import cv2
import storage

drawing = False   # true if mouse is pressed
cArray = []
fNo = 0   # stores the frame number

def findObject(frameNum, x, y,):
    from numpy.linalg.linalg import inv
    from numpy import loadtxt
    global width, height
    homographyFilename = "laurier-homography.txt"
    homography = inv(loadtxt(homographyFilename))
    databaseFilename = "laurier.sqlite"
    trajectoryType = "object"
    objects = storage.loadTrajectoriesFromSqlite(databaseFilename, trajectoryType)
    features = storage.loadTrajectoriesFromSqlite(databaseFilename, "feature")
    px = 0.2
    py = 0.2
    pixelThreshold = 800
    for obj in objects:
        if obj.existsAtInstant(frameNum):
            obj.setFeatures(features)
            if obj.hasFeatures():
                u = []
                v = []
                for f in obj.getFeatures():
                    if f.existsAtInstant(frameNum):
                        projectedPosition = f.getPositionAtInstant(frameNum).project(homography)
                        u.append(projectedPosition.x)
                        v.append(projectedPosition.y)
                xmin = min(u)
                xmax = max(u)
                ymin = min(v)
                ymax = max(v)
                xMm = px * (xmax - xmin)
                yMm = py * (ymax - ymin)
                a = max(ymax - ymin + (2 * yMm), xmax - (xmin + 2 * xMm))
                yCropMin = int(max(0, .5 * (ymin + ymax - a)))
                yCropMax = int(min(height - 1, .5 * (ymin + ymax + a)))
                xCropMin = int(max(0, .5 * (xmin + xmax - a)))
                xCropMax = int(min(width - 1, .5 * (xmin + xmax + a)))
                if yCropMax != yCropMin and xCropMax != xCropMin and (yCropMax - yCropMin) * (xCropMax - xCropMin) > pixelThreshold:
                    if x > xCropMin and x < xCropMax and y > yCropMin and y < yCropMax:
                        print "object detected: " + format(obj.getNum())

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
    for i in range(len(cArray)):
        cv2.circle(frame, (cArray[i][0], cArray[i][1]), 3, (0, 255, 0), -1)
    cv2.imshow('Video', frame)
    fNo += 1
    if cv2.waitKey(75) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
