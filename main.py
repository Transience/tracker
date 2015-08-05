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
cArray = []   # stores new trajectory positions (temporary) in order to display the trace
fNo = 0   # stores the frame number
objTag = None   # stores selected object's id
track = False   # turns on track mode if true
merge = False   # turns on merge mode if true
split = False   # turns on split mode if true
mergeList = []   # holds the id of objects to be merged
trace = []   # holds the trace coordinates
splitSelect = []   # holds trajectory ids selected for splitting
pace = 0   # to adjust the video speed

def findObject(frameNum, x=None, y=None):   # finds the object clicked on (utilizes original sqlite)
    global objects, features, objTag, merge, mergeList
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
            if xmax > x > xmin and ymax > y > ymin:
                print "object detected: " + format(obj.getNum())
                print "object position: " + format(obj.getPositionAtInstant(frameNum).project(homography))
                objTag = obj.getNum()
                if merge is True:
                    mergeList.append(obj.getNum())
    return box   # returns pixel range for each object

def findTrajectory(frameNum):   # finds the features selected by the user
    global cObjects, features, cArray, splitSelect
    for obj in cObjects:
        if obj.existsAtInstant(frameNum):
            cObjFeatures = [features[i] for i in obj.featureNumbers]
            for cObjFeature in cObjFeatures:
                if cObjFeature.existsAtInstant(frameNum):
                    for f in range(cObjFeature.getFirstInstant(), frameNum):
                        position = cObjFeature.getPositionAtInstant(f).project(homography)
                        for coord in cArray:
                            if coord[0]+5 > position[0] > coord[0]-5 and coord[1]+5 > position[1] > coord[1]-5:
                                if not cObjFeature.getNum() in splitSelect:
                                    splitSelect.append(cObjFeature.getNum())

def drawTrajectory(frame, frameNum):   # draws trajectory for each object
    global cObjects, features, splitSelect
    if split is False:
        for obj in cObjects:
            if obj.existsAtInstant(frameNum):
                prevPosition = obj.getPositionAtInstant(obj.getFirstInstant()).project(homography)
                for f in range(obj.getFirstInstant(), frameNum):
                    position = obj.getPositionAtInstant(f).project(homography)
                    cv2.line(frame, (position[0], position[1]), (prevPosition[0], prevPosition[1]), (0, 0, 255), 2)
                    prevPosition = position
    else:
        for obj in cObjects:
            if obj.existsAtInstant(frameNum):
                cObjFeatures = [features[i] for i in obj.featureNumbers]
                for cObjFeature in cObjFeatures:
                    if cObjFeature.existsAtInstant(frameNum):
                        if cObjFeature.getNum() in splitSelect:
                            prevPosition = cObjFeature.getPositionAtInstant(cObjFeature.getFirstInstant()).project(homography)
                            for f in range(cObjFeature.getFirstInstant(), frameNum):
                                position = cObjFeature.getPositionAtInstant(f).project(homography)
                                cv2.line(frame, (position[0], position[1]), (prevPosition[0], prevPosition[1]), (0, 255, 255), 1)
                                prevPosition = position
                        else:
                            prevPosition = cObjFeature.getPositionAtInstant(cObjFeature.getFirstInstant()).project(homography)
                            for f in range(cObjFeature.getFirstInstant(), frameNum):
                                position = cObjFeature.getPositionAtInstant(f).project(homography)
                                cv2.line(frame, (position[0], position[1]), (prevPosition[0], prevPosition[1]), (0, 0, 255), 1)
                                prevPosition = position

def drawBox(frame, frameNum):   # annotates each object and highlights when clicked
    global objTag
    if split is False:
        box = findObject(frameNum)
        for i in range(len(box)):
            if box[i][4] == objTag:
                cv2.rectangle(frame, (box[i][3], box[i][0]), (box[i][2], box[i][1]), (0, 255, 255), 3)
            else:
                cv2.rectangle(frame, (box[i][3], box[i][0]), (box[i][2], box[i][1]), (255, 0, 0), 3)

def drawEditBox(frame):   # for the static text
    global width, height
    if track is True:
        cv2.rectangle(frame, (0, 0), (width, height), (0, 255, 255), 3)
        cv2.putText(frame,"track mode", (width-100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
    else:
        cv2.putText(frame,"toggle track (t)", (width-130, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
    if merge is True:
        cv2.rectangle(frame, (0, 0), (width, height), (0, 255, 0), 3)
        cv2.putText(frame,"merge mode", (width-125, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
    else:
        cv2.putText(frame,"toggle merge (m)", (width-150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
    if split is True:
        cv2.rectangle(frame, (0, 0), (width, height), (255, 0, 0), 3)
        cv2.putText(frame,"split mode", (width-125, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
    else:
        cv2.putText(frame,"toggle split (s)", (width-125, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
    cv2.putText(frame,"reset edits (r)", (width-125, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
    cv2.putText(frame,"video speed (0-4)", (width-150, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))

def sqlSplit(newObjID):   # splits an object into two 
    global splitSelect, cObjects
    try:
        connection = sqlite3.connect(newFilename)
        cursor = connection.cursor()
        cursor.execute("SELECT object_id from objects_features where trajectory_id = " + format(splitSelect[0]) + ";")
        objID = cursor.fetchone()[0]
        cursor.execute("SELECT * from objects where object_id = " + format(objID) + ";")
        data = cursor.fetchone()
        cursor.execute("insert into objects (object_id, road_user_type, n_objects) values (?, ?, ?);", (len(cObjects), data[1], data[2]))
        sql = "update objects_features set object_id = " + format(newObjID) + " where object_id = " + format(objID) + " and trajectory_id in (?"
        extension = ''.join(itertools.repeat(', ?', len(splitSelect)-1))
        sql = sql + extension + ");"
        cursor.execute(sql, splitSelect)
        del splitSelect[:]
        connection.commit()
        connection.close()
    except sqlite3.Error, e:
            print "Error %s:" % e.args[0]

def sqlMerge():   # merges two or more objects selected by the user
    global mergeList, cObjects
    frameRange = []   # to store the first instant and last instant of the objects to be merged
    if len(mergeList)>1:
        try:
            connection = sqlite3.connect(newFilename)
            cursor = connection.cursor()
            for i in range(1, len(mergeList)):
                for obj in cObjects:
                    if obj.getNum() == mergeList[i]:
                        cursor.execute("delete from objects where object_id = " + format(mergeList[i]) + ";")
                        cursor.execute("update objects_features set object_id = " + format(mergeList[0]) + " where object_id = " + format(mergeList[i]) + ";")
            for i in range(len(mergeList)):
                for obj in cObjects:
                    if obj.getNum() == mergeList[i]:
                        frameRange.append([obj.getFirstInstant(), obj.getLastInstant(), obj.getNum()])
            frameRange = sorted(frameRange)
            for i in range(len(frameRange)-1):
                if frameRange[i][1] < frameRange[i+1][0]:   # looks for discontinuity
                    for obj in cObjects:
                        if obj.getNum() == frameRange[i][2]:
                            position = obj.getPositionAtInstant(frameRange[i][1])
                            cursor.execute("SELECT max(trajectory_id) from positions where trajectory_id in (select trajectory_id from objects_features where object_id = "
                                           + format(mergeList[0]) + ") and frame_number = " + format(frameRange[i][1]) + ";")
                            tID = cursor.fetchone()[0]
                            for f in range(frameRange[i][1]+1, frameRange[i+1][0]):
                                cursor.execute("insert into positions (trajectory_id, frame_number, x_coordinate, y_coordinate) values (?, ?, ?, ?);", (tID, f, position[0], position[1]))
            connection.commit()
            connection.close()
            del mergeList[:]
        except sqlite3.Error, e:
            print "Error %s:" % e.args[0]

def sqlTrack(objID, frames, coords):   # performs delete and insert operations on the sqlite (new file)
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
        f = frames[0]
        for i in range(len(frames)):
            jump = frames[i] - f
            if 5 > jump > 1:
                c = [(coords[i][0] + coords[i-1][0])/2, (coords[i][1] + coords[i-1][1])/2]
                for k in range(f+1, frames[i]):
                    cursor.execute("delete from positions where trajectory_id in (select trajectory_id from objects_features where object_id = " + format(objID) + ") and frame_number = " + format(k) + ";")
                    cursor.execute("insert into positions (trajectory_id, frame_number, x_coordinate, y_coordinate) values (?, ?, ?, ?);", (tID, k, c[0], c[1]))
            f = frames[i]
            cursor.execute("insert into positions (trajectory_id, frame_number, x_coordinate, y_coordinate) values (?, ?, ?, ?);", (tID, frames[i], coords[i][0], coords[i][1]))
        connection.commit()
        connection.close()
    except sqlite3.Error, e:
        print "Error %s:" % e.args[0]

def tracing():   # extract data from the trace array, removing redundant data for a single frame
    global trace
    frames = []
    coords = []
    tempF = None
    temp = None
    for record in trace:
        if not temp == record[0]:
            if not len(frames) == 0:
                sqlTrack(temp, frames, coords)
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
    sqlTrack(temp, frames, coords)

def coordinates(event, x, y, flags, param):
    global drawing, cArray, fNo, objTag, trace
    if event == cv2.EVENT_LBUTTONDOWN:
        print x, y
        drawing = True
        cArray.append([x, y])
        findObject(fNo, x, y)
        if objTag is not None and track == True:
                trace.append([objTag, fNo, x, y])
                print "tracing object: " + format(objTag) + " (" + format(x) + " ," + format(y) + ")"
        if split is True:
            findTrajectory(fNo)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cArray.append([x, y])
            if objTag is not None and track == True:
                trace.append([objTag, fNo, x, y])
                print "tracing object: " + format(objTag) + " (" + format(x) + " ," + format(y) + ")"
    elif event == cv2.EVENT_LBUTTONUP:
        objTag = None   # deselects the object
        drawing = False
        del cArray[:]

cap = cv2.VideoCapture(videoFilename)
cv2.namedWindow('Video')
cv2.setMouseCallback('Video', coordinates)
width = int(cap.get(3))
height = int(cap.get(4))
newObjID = len(cObjects)

while(cap.isOpened()):
    ret, frame = cap.read()
    drawBox(frame, fNo)
    drawTrajectory(frame, fNo)
    drawEditBox(frame)
    if split is True:
        findTrajectory(fNo)
    for i in range(len(cArray)-1):   # displays the user drawn trajectory
        cv2.line(frame, (cArray[i][0], cArray[i][1]), (cArray[i+1][0], cArray[i+1][1]), (0, 255, 0), 2)
    cv2.imshow('Video', frame)
    fNo += 1
    k = cv2.waitKey(pace) & 0xFF   # set cv2.waitKey(0) for frame by frame editing
    if k == 27:   # exit with committing the trace
        if trace:
            tracing()
        break
    elif k == 116:   # toggle track mode
        track = track != True
        merge = False
        split = False
    elif k == 115:   # toggle split mode
        split = split != True
        if split is False:   # calling sqlSplit() while coming out of merge mode
            if splitSelect:
                sqlSplit(newObjID)
                newObjID += 1
        track = False
        merge = False
    elif k == 109:   # toggle merge mode
        merge = merge != True
        if merge is False:   # calling sqlMerge() while coming out of merge mode
            sqlMerge()
        split = False
        track = False
    elif k == 114:   # creates a copy of the original sqlite
        shutil.copy2(databaseFilename, newFilename)
    elif k == 48:
        pace = 0
    elif k == 49:
        pace = 150
    elif k == 50:
        pace = 100
    elif k == 51:
        pace = 50
    elif k == 52:
        pace = 25

cap.release()
cv2.destroyAllWindows()
