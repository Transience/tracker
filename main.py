__author__ = 'Akhil'

import cv2
import storage
import sqlite3
import cvutils
import itertools
import shutil
from numpy.linalg.linalg import inv
from numpy import loadtxt

homographyFilename = "laurier-homography.txt"
homography = inv(loadtxt(homographyFilename))
databaseFilename = "laurier.sqlite"
newFilename = "corrected.sqlite"
videoFilename = "laurier.avi"
cObjects = storage.loadTrajectoriesFromSqlite(newFilename, "object")
objects = storage.loadTrajectoriesFromSqlite(databaseFilename, "object")
features = storage.loadTrajectoriesFromSqlite(databaseFilename, "feature")

drawing = False   # true if mouse is pressed
cArray = []   # stores new trajectory positions (temporary) inorder to display the trace
fNo = 0   # stores the frame number
objTag = None   # stores selected object's id
edit = False   # turns on edit mode if true
trace = []   # holds the trace coordinates

def findObject(frameNum, x=None, y=None):   # finds the object clicked on
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
            if x is None and y is None:   # utilized when the function call is from drawBox()
                box.append([ymax, ymin, xmax, xmin, obj.getNum()])
            if obj.getNum() == objTag and obj.getLastInstant() == frameNum:   # deselect the object when out of range
                objTag = None
            if xmax > x > xmin and ymax > y > ymin:
                print "object detected: " + format(obj.getNum())
                print "object position: " + format(obj.getPositionAtInstant(frameNum).project(homography))
                objTag = obj.getNum()
    return box

def drawTrajectory(frame, frameNum):   # draws trajectory for each object (utilizes original sqlite)
    global cObjects
    for obj in cObjects:
        if obj.existsAtInstant(frameNum):
            prevPosition = obj.getPositionAtInstant(obj.getFirstInstant()).project(homography)
            for f in range(obj.getFirstInstant(), frameNum):
                position = obj.getPositionAtInstant(f).project(homography)
                cv2.line(frame, (position[0], position[1]), (prevPosition[0], prevPosition[1]), (0, 0, 255), 2)
                prevPosition = position

def drawBox(frame, frameNum):   # annotates each object and highlights when clicked
    global objTag
    box = findObject(frameNum)
    for i in range(len(box)):
        if box[i][4] == objTag:
            cv2.rectangle(frame, (box[i][3], box[i][0]), (box[i][2], box[i][1]), (0, 255, 255), 3)
        else:
            cv2.rectangle(frame, (box[i][3], box[i][0]), (box[i][2], box[i][1]), (255, 0, 0), 3)

def drawEditBox(frame):   # for the static text
    global width, height, edit
    if edit == True:
        cv2.rectangle(frame, (0, 0), (width, height), (0, 255, 255), 3)
        cv2.putText(frame,"edit mode", (width-100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
    else:
        cv2.putText(frame,"toggle edit (e)", (width-125, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
    cv2.putText(frame,"reset edits (r)", (width-125, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
    cv2.putText(frame,"deselect(rClick)", (width-125, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))

def sqlEdit(objID, frames, coords):   # performs delete and insert operations on the sqlite (new file)
    try:
        connection = sqlite3.connect(newFilename)
        cursor = connection.cursor()
        extension = ''.join(itertools.repeat(', ?', len(frames)-1))
        sql = "select min(trajectory_id) from positions where trajectory_id in (select trajectory_id from objects_features where object_id = " + format(objID) + ") and frame_number in (?"
        sql2 = "delete from positions where trajectory_id in (select trajectory_id from objects_features where object_id = " + format(objID) + ") and frame_number in (?"
        sql = sql + extension + ");"
        sql2 = sql2 + extension + ");"
        cursor.execute(sql, frames)
        tID = cursor.fetchone()[0]   # tID will be the trajectory id of the new feature
        cursor.execute(sql2, frames)
        for i in range(len(frames)):
            cursor.execute("insert into positions (trajectory_id, frame_number, x_coordinate, y_coordinate) values (?, ?, ?, ?);", (tID, frames[i], coords[i][0], coords[i][1]))
        connection.commit()
        connection.close()
    except sqlite3.Error, e:
        print "Error %s:" % e.args[0]

def tracing():   # extract data from the trace array, removing redundant data for a single frame
    global trace
    print trace[0][1], trace[len(trace)-1][1]
    frames = []
    coords = []
    tempF = None
    temp = None
    for record in trace:
        if not temp == record[0]:
            if not len(frames) == 0:
                sqlEdit(temp, frames, coords)
                del frames[:]
                del coords[:]
            temp = record[0]
        if not tempF == record[1]:
            tempF = record[1]
            frames.append(tempF)
            point = [record[2], record[3]]
            invH = cvutils.invertHomography(homography)
            coord = cvutils.project(invH, point)
            coords.append([coord[0][0], coord[1][0]])
    sqlEdit(temp, frames, coords)

def coordinates(event, x, y, flags, param):
    global drawing, cArray, fNo, objTag, trace
    if event == cv2.EVENT_LBUTTONDOWN:
        print x, y
        drawing = True
        cArray.append([x, y])
        findObject(fNo, x, y)
        if objTag is not None and edit == True:   # tracing commences
                trace.append([objTag, fNo, x, y])
                print "editing object: " + format(objTag) + " (" + format(x) + " ," + format(y) + ")"
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cArray.append([x, y])
            if objTag is not None and edit == True:   # tracing continues
                trace.append([objTag, fNo, x, y])
                print "editing object: " + format(objTag) + " (" + format(x) + " ," + format(y) + ")"
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        del cArray[:]
    if event == cv2.EVENT_RBUTTONDOWN:   # deselect a selected object
        objTag = None

cap = cv2.VideoCapture(videoFilename)
cv2.namedWindow('Video')
cv2.setMouseCallback('Video', coordinates)
width = int(cap.get(3))
height = int(cap.get(4))

while(cap.isOpened()):
    ret, frame = cap.read()
    drawBox(frame, fNo)
    drawTrajectory(frame, fNo)
    drawEditBox(frame)
    for i in range(len(cArray)):   # displays the user drawn trajectory
        cv2.circle(frame, (cArray[i][0], cArray[i][1]), 3, (0, 255, 0), -1)
    cv2.imshow('Video', frame)
    fNo += 1
    k = cv2.waitKey(0) & 0xFF
    if k == 27:   # exit with committing the trace
        if trace:
            tracing()
        break
    elif k == 101:   # toggle edit mode
        edit = edit != True
    elif k == 114:   # creates a copy of the original sqlite
        shutil.copy2(databaseFilename, newFilename)

cap.release()
cv2.destroyAllWindows()
