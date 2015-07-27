__author__ = 'Akhil'

import cv2
import storage
import sqlite3
import cvutils
from numpy.linalg.linalg import inv
from numpy import loadtxt

homographyFilename = "laurier-homography.txt"
homography = inv(loadtxt(homographyFilename))
databaseFilename = "laurier.sqlite"
newFilename = "corrected.sqlite"
objects = storage.loadTrajectoriesFromSqlite(databaseFilename, "object")
features = storage.loadTrajectoriesFromSqlite(databaseFilename, "feature")

drawing = False   # true if mouse is pressed
cArray = []   # stores new trajectory positions (temporary)
fNo = 0   # stores the frame number
objTag = None   # stores selected object's id
edit = False   # turns on edit mode if true
trace = []   # holds the trace coordinates

def findObject(frameNum, x=None, y=None):
    global objects, features, objTag
    box = []
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
            else:
                objTag = None
            if obj.getNum() == objTag and obj.getLastInstant() == frameNum:
                objTag = None
            if xmax > x > xmin and ymax > y > ymin:
                print "object detected: " + format(obj.getNum())
                print "object position: " + format(obj.getPositionAtInstant(frameNum).project(homography))
                objTag = obj.getNum()
    return box

def drawTrajectory(frame, frameNum):
    global objects, editedObjects
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
            cv2.rectangle(frame, (box[i][3], box[i][0]), (box[i][2], box[i][1]), (0, 255, 255), 3)
        else:
            cv2.rectangle(frame, (box[i][3], box[i][0]), (box[i][2], box[i][1]), (255, 0, 0), 3)

def drawEditBox(frame):
    global width, height, edit
    if edit == True:
        cv2.rectangle(frame, (0, 0), (width, height), (0, 255, 255), 3)
        cv2.putText(frame,"edit mode", (width-100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
    else:
        cv2.putText(frame,"toggle edit (e)", (width-125, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))

def coordinates(event, x, y, flags, param):
    global drawing, cArray, fNo, objTag, trace
    if event == cv2.EVENT_LBUTTONDOWN:
        print x, y
        drawing = True
        cArray.append([x, y])
        findObject(fNo, x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cArray.append([x, y])
            if objTag is not None and edit == True:
                trace.append([objTag, fNo, x, y])
                print "editing object: " + format(objTag) + " (" + format(x) + " ," + format(y) + ")"
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if objTag is not None and edit == True:
            addFeatures()
        del cArray[:]

def addFeatures():
    global objTag
    try:
        connection = sqlite3.connect(databaseFilename)
        newConnection = sqlite3.connect(newFilename)
        cursor = connection.cursor()
        newCursor = newConnection.cursor()
        cursor.execute("SELECT * from objects where object_id = " + format(objTag) + ";")
        objData = cursor.fetchone()
        newCursor.execute("insert or replace into objects (object_id, road_user_type, n_objects) values (?, ?, ?);", objData)
        newCursor.execute("insert or replace into objects_features (object_id, trajectory_id) values (?, ?);", (objTag, objTag))
        newConnection.commit()
        connection.close()
        newConnection.close()
    except sqlite3.Error, e:
        print "Error %s:" % e.args[0]

def tracing():
    global trace
    try:
        connection = sqlite3.connect(newFilename)
        cursor = connection.cursor()
        for record in trace:
            point = [record[2], record[3]]
            invH = cvutils.invertHomography(homography)
            coords = cvutils.project(invH, point)
            cursor.execute("insert or replace into positions (trajectory_id, frame_number, x_coordinate, y_coordinate) values (?, ?, ?, ?);", (record[0], record[1], coords[0][0], coords[1][0]))
        connection.commit()
        connection.close()
    except sqlite3.Error, e:
        print "Error %s:" % e.args[0]

def createDatabase():
    try:
        connection = sqlite3.connect(newFilename)
        cursor = connection.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS objects (object_id INTEGER, road_user_type INTEGER DEFAULT 0, n_objects INTEGER DEFAULT 1, PRIMARY KEY(object_id))")
        cursor.execute("CREATE TABLE IF NOT EXISTS objects_features (object_id INTEGER, trajectory_id INTEGER, PRIMARY KEY(object_id, trajectory_id))")
        cursor.execute("CREATE TABLE IF NOT EXISTS positions (trajectory_id INTEGER,frame_number INTEGER, x_coordinate REAL, y_coordinate REAL, PRIMARY KEY(trajectory_id, frame_number))")
        cursor.execute("CREATE TABLE IF NOT EXISTS velocities (trajectory_id INTEGER,frame_number INTEGER, x_coordinate REAL, y_coordinate REAL, PRIMARY KEY(trajectory_id, frame_number))")
        connection.close()
    except sqlite3.Error, e:
        print "Error %s:" % e.args[0]

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
        tracing()
        break
    if k == 101:
        edit = edit != True
        if edit == True:
            createDatabase()
    for i in range(len(cArray)):
        cv2.circle(frame, (cArray[i][0], cArray[i][1]), 3, (0, 255, 0), -1)
    cv2.imshow('Video', frame)
    fNo += 1

cap.release()
cv2.destroyAllWindows()
