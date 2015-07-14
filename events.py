#! /usr/bin/env python
'''Libraries for events
Interactions, pedestrian crossing...'''

import numpy as np
from numpy import arccos

import multiprocessing
import itertools

import moving, prediction, indicators, utils, cvutils
__metaclass__ = type

def findRoute(prototypes,objects,i,j,noiseEntryNums,noiseExitNums,minSimilarity= 0.3, spatialThreshold=1.0, delta=180):
    if i[0] not in noiseEntryNums: 
        prototypesRoutes= [ x for x in sorted(prototypes.keys()) if i[0]==x[0]]
    elif i[1] not in noiseExitNums:
        prototypesRoutes=[ x for x in sorted(prototypes.keys()) if i[1]==x[1]]
    else:
        prototypesRoutes=[x for x in sorted(prototypes.keys())]
    routeSim={}
    lcss = utils.LCSS(similarityFunc=lambda x,y: (distanceForLCSS(x,y) <= spatialThreshold),delta=delta)
    for y in prototypesRoutes: 
        if y in prototypes.keys():
            prototypesIDs=prototypes[y]
            similarity=[]
            for x in prototypesIDs:
                s=lcss.computeNormalized(objects[j].positions, objects[x].positions)
                similarity.append(s)
            routeSim[y]=max(similarity)
    route=max(routeSim, key=routeSim.get)
    if routeSim[route]>=minSimilarity:
        return route
    else:
        return i

def getRoute(obj,prototypes,objects,noiseEntryNums,noiseExitNums,useDestination=True):
    route=(obj.startRouteID,obj.endRouteID)
    if useDestination:
        if route not in prototypes.keys():
            route= findRoute(prototypes,objects,route,obj.getNum(),noiseEntryNums,noiseExitNums)
    return route

class Interaction(moving.STObject):
    '''Class for an interaction between two road users 
    or a road user and an obstacle
    
    link to the moving objects
    contains the indicators in a dictionary with the names as keys
    '''

    categories = {'Head On': 0,
                  'rearend': 1,
                  'side': 2,
                  'parallel': 3}

    indicatorNames = ['Collision Course Dot Product',
                      'Collision Course Angle',
                      'Distance',
                      'Minimum Distance',
                      'Velocity Angle',
                      'Speed Differential',
                      'Collision Probability',
                      'Time to Collision', # 7
                      'Probability of Successful Evasive Action',
                      'predicted Post Encroachment Time']

    indicatorNameToIndices = utils.inverseEnumeration(indicatorNames)

    indicatorShortNames = ['CCDP',
                           'CCA',
                           'Dist',
                           'MinDist',
                           'VA',
                           'SD',
                           'PoC',
                           'TTC',
                           'P(SEA)',
                           'pPET']

    indicatorUnits = ['',
                      'rad',
                      'm',
                      'm',
                      'rad',
                      'm/s',
                      '',
                      's',
                      '',
                      '']

    def __init__(self, num = None, timeInterval = None, roaduserNum1 = None, roaduserNum2 = None, roadUser1 = None, roadUser2 = None, categoryNum = None):
        moving.STObject.__init__(self, num, timeInterval)
        if timeInterval is None and roadUser1 is not None and roadUser2 is not None:
            self.timeInterval = roadUser1.commonTimeInterval(roadUser2)
        self.roadUser1 = roadUser1
        self.roadUser2 = roadUser2
        if roaduserNum1 is not None and roaduserNum2 is not None:
            self.roadUserNumbers = set([roaduserNum1, roaduserNum2])
        elif roadUser1 is not None and roadUser2 is not None:
            self.roadUserNumbers = set([roadUser1.getNum(), roadUser2.getNum()])
        else:
            self.roadUserNumbers = None
        self.categoryNum = categoryNum
        self.indicators = {}
        self.interactionInterval = None
         # list for collison points and crossing zones
        self.collisionPoints = None
        self.crossingZones = None

    def getRoadUserNumbers(self):
        return self.roadUserNumbers

    def setRoadUsers(self, objects):
        nums = list(self.getRoadUserNumbers())
        if objects[nums[0]].getNum() == nums[0]:
            self.roadUser1 = objects[nums[0]]
        if objects[nums[1]].getNum() == nums[1]:
            self.roadUser2 = objects[nums[1]]

        i = 0
        while i < len(objects) and self.roadUser2 is None:
            if objects[i].getNum() in nums:
                if self.roadUser1 is None:
                    self.roadUser1 = objects[i]
                else:
                    self.roadUser2 = objects[i]
            i += 1

    def getIndicator(self, indicatorName):
        return self.indicators.get(indicatorName, None)

    def addIndicator(self, indicator):
        if indicator is not None:
            self.indicators[indicator.name] = indicator

    def getIndicatorValueAtInstant(self, indicatorName, instant):
        indicator = self.getIndicator(indicatorName)
        if indicator is not None:
            return indicator[instant]
        else:
            return None

    def getIndicatorValuesAtInstant(self, instant):
        '''Returns list of indicator values at instant
        as dict (with keys from indicators dict)'''
        values = {}
        for k, indicator in self.indicators.iteritems():
            values[k] = indicator[instant]
        return values
        
    def plot(self, options = '', withOrigin = False, timeStep = 1, withFeatures = False, **kwargs):
        self.roadUser1.plot(options, withOrigin, timeStep, withFeatures, **kwargs)
        self.roadUser2.plot(options, withOrigin, timeStep, withFeatures, **kwargs)

    def plotOnWorldImage(self, nPixelsPerUnitDistance, options = '', withOrigin = False, timeStep = 1, **kwargs):
        self.roadUser1.plotOnWorldImage(nPixelsPerUnitDistance, options, withOrigin, timeStep, **kwargs)
        self.roadUser2.plotOnWorldImage(nPixelsPerUnitDistance, options, withOrigin, timeStep, **kwargs)

    def play(self, videoFilename, homography = None, undistort = False, intrinsicCameraMatrix = None, distortionCoefficients = None, undistortedImageMultiplication = 1.):
        if self.roadUser1 is not None and self.roadUser2 is not None:
            cvutils.displayTrajectories(videoFilename, [self.roadUser1, self.roadUser2], homography = homography, firstFrameNum = self.getFirstInstant(), lastFrameNumArg = self.getLastInstant(), undistort = undistort, intrinsicCameraMatrix = intrinsicCameraMatrix, distortionCoefficients = distortionCoefficients, undistortedImageMultiplication = undistortedImageMultiplication)
        else:
            print('Please set the interaction road user attributes roadUser1 and roadUser1 through the method setRoadUsers')

    def computeIndicators(self):
        '''Computes the collision course cosine only if the cosine is positive'''
        collisionCourseDotProducts = {}#[0]*int(self.timeInterval.length())
        collisionCourseAngles = {}
        velocityAngles = {}
        distances = {}#[0]*int(self.timeInterval.length())
        speedDifferentials = {}
        interactionInstants = []
        for instant in self.timeInterval:
            deltap = self.roadUser1.getPositionAtInstant(instant)-self.roadUser2.getPositionAtInstant(instant)
            v1 = self.roadUser1.getVelocityAtInstant(instant)
            v2 = self.roadUser2.getVelocityAtInstant(instant)
            deltav = v2-v1
            velocityAngles[instant] = arccos(moving.Point.dot(v1, v2)/(v1.norm2()*v2.norm2()))
            collisionCourseDotProducts[instant] = moving.Point.dot(deltap, deltav)
            distances[instant] = deltap.norm2()
            speedDifferentials[instant] = deltav.norm2()
            if collisionCourseDotProducts[instant] > 0:
                interactionInstants.append(instant)
            if distances[instant] != 0 and speedDifferentials[instant] != 0:
                collisionCourseAngles[instant] = arccos(collisionCourseDotProducts[instant]/(distances[instant]*speedDifferentials[instant]))

        if len(interactionInstants) >= 2:
            self.interactionInterval = moving.TimeInterval(interactionInstants[0], interactionInstants[-1])
        else:
            self.interactionInterval = moving.TimeInterval()
        self.addIndicator(indicators.SeverityIndicator(Interaction.indicatorNames[0], collisionCourseDotProducts))
        self.addIndicator(indicators.SeverityIndicator(Interaction.indicatorNames[1], collisionCourseAngles))
        self.addIndicator(indicators.SeverityIndicator(Interaction.indicatorNames[2], distances))
        self.addIndicator(indicators.SeverityIndicator(Interaction.indicatorNames[4], velocityAngles))
        self.addIndicator(indicators.SeverityIndicator(Interaction.indicatorNames[5], speedDifferentials))

        # if we have features, compute other indicators
        if self.roadUser1.hasFeatures() and self.roadUser2.hasFeatures():
            minDistance={}
            for instant in self.timeInterval:
                minDistance[instant] = moving.MovingObject.minDistance(self.roadUser1, self.roadUser2, instant)
            self.addIndicator(indicators.SeverityIndicator(Interaction.indicatorNames[3], minDistance))

    def computeCrossingsCollisions(self, predictionParameters, collisionDistanceThreshold, timeHorizon, computeCZ = False, debug = False, timeInterval = None, nProcesses = 1, usePrototypes=False, route1= (-1,-1), route2=(-1,-1), prototypes={}, secondStepPrototypes={}, nMatching={}, objects=[], noiseEntryNums=[], noiseExitNums=[], minSimilarity=0.1, mostMatched=None, useDestination=True, useSpeedPrototype=True, acceptPartialLength=30, step=1):
        '''Computes all crossing and collision points at each common instant for two road users. '''
        TTCs = {}
        if usePrototypes:
            route1= getRoute(self.roadUser1,prototypes,objects,noiseEntryNums,noiseExitNums,useDestination)
            route2= getRoute(self.roadUser2,prototypes,objects,noiseEntryNums,noiseExitNums,useDestination)

        if timeInterval is not None:
            commonTimeInterval = timeInterval
        else:
            commonTimeInterval = self.timeInterval
        self.collisionPoints, crossingZones = predictionParameters.computeCrossingsCollisions(self.roadUser1, self.roadUser2, collisionDistanceThreshold, timeHorizon, computeCZ, debug, commonTimeInterval, nProcesses,usePrototypes,route1,route2,prototypes,secondStepPrototypes,nMatching,objects,noiseEntryNums,noiseExitNums,minSimilarity,mostMatched,useDestination,useSpeedPrototype,acceptPartialLength, step)
        for i, cp in self.collisionPoints.iteritems():
            TTCs[i] = prediction.SafetyPoint.computeExpectedIndicator(cp)
        self.addIndicator(indicators.SeverityIndicator(Interaction.indicatorNames[7], TTCs, mostSevereIsMax=False))
        
        # crossing zones and pPET
        if computeCZ:
            self.crossingZones[predictionParameters.name] = crossingZones
            pPETs = {}
            for i, cz in self.crossingZones.iteritems():
                pPETs[i] = prediction.SafetyPoint.computeExpectedIndicator(cz)
            self.addIndicator(indicators.SeverityIndicator(Interaction.indicatorNames[9], pPETs, mostSevereIsMax=False))
        # TODO add probability of collision, and probability of successful evasive action

    def computePET(self, collisionDistanceThreshold):
        # TODO add crossing zone
        self.pet = moving.MovingObject.computePET(self.roadUser1, self.roadUser2, collisionDistanceThreshold)

    def addVideoFilename(self,videoFilename):
        self.videoFilename = videoFilename

    def addInteractionType(self,interactionType):
        ''' interaction types: conflict or collision if they are known'''
        self.interactionType = interactionType

    def getCrossingZones(self, predictionMethodName):
        if self.crossingZones is not None:
            return self.crossingZones[predictionMethodName]
        else:
            return None

    def getCollisionPoints(self, predictionMethodName):
        if self.collisionPoints is not None:
            return self.collisionPoints[predictionMethodName]
        else:
            return None


def createInteractions(objects, _others = None):
    '''Create all interactions of two co-existing road users'''
    if _others is not None:
        others = _others

    interactions = []
    num = 0
    for i in xrange(len(objects)):
        if _others is None:
            others = objects[:i]
        for j in xrange(len(others)):
            commonTimeInterval = objects[i].commonTimeInterval(others[j])
            if not commonTimeInterval.empty():
                interactions.append(Interaction(num, commonTimeInterval, objects[i].num, others[j].num, objects[i], others[j]))
                num += 1
    return interactions

def findInteraction(interactions, roadUserNum1, roadUserNum2):
    'Returns the right interaction in the set'
    i=0
    while i<len(interactions) and set([roadUserNum1, roadUserNum2]) != interactions[i].getRoadUserNumbers():
        i+=1
    if i<len(interactions):
        return interactions[i]
    else:
        return None

def aggregateSafetyPoints(interactions, predictionMethodName = None, pointType = 'collision'):
    '''Put all collision points or crossing zones in a list for display'''
    if predictionMethodName is None and len(interactions)>0:
        predictionMethodName = interactions[0].collisionPoints.keys()[0]

    allPoints = []
    if pointType == 'collision':
        for i in interactions:
            for points in i.collisionPoints[predictionMethodName].values():
                allPoints += points
    elif pointType == 'crossing':
        for i in interactions:
            for points in i.crossingZones[predictionMethodName].values():
                allPoints += points
    else:
        print('unknown type of point '+pointType)
    return allPoints

def prototypeCluster(interactions, similarityMatrix, alignmentMatrix, indicatorName, minSimilarity):
    '''Finds exemplar indicator time series for all interactions
    Returns the prototype indices (in the interaction list) and the label of each indicator (interaction)

    if an indicator profile (time series) is different enough (<minSimilarity), 
    it will become a new prototype. 
    Non-prototype interactions will be assigned to an existing prototype'''

    # sort indicators based on length
    indices = range(similarityMatrix.shape[0])
    def compare(i, j):
        if len(interactions[i].getIndicator(indicatorName)) > len(interactions[j].getIndicator(indicatorName)):
            return -1
        elif len(interactions[i].getIndicator(indicatorName)) == len(interactions[j].getIndicator(indicatorName)):
            return 0
        else:
            return 1
    indices.sort(compare)
    # go through all indicators
    prototypeIndices = [indices[0]]
    for i in indices[1:]:
        if similarityMatrix[i][prototypeIndices].max() < minSimilarity:
             prototypeIndices.append(i)

    # assignment
    labels = [-1]*similarityMatrix.shape[0]
    indices = [i for i in range(similarityMatrix.shape[0]) if i not in prototypeIndices]
    for i in prototypeIndices:
        labels[i] = i
    for i in indices:
        prototypeIndex = similarityMatrix[i][prototypeIndices].argmax()
        labels[i] = prototypeIndices[prototypeIndex]

    return prototypeIndices, labels

def prototypeMultivariateCluster(interactions, similarityMatrics, indicatorNames, minSimilarities, minClusterSize):
    '''Finds exmaple indicator time series (several indicators) for all interactions

    if any interaction indicator time series is different enough (<minSimilarity),
    it will become a new prototype. 
    Non-prototype interactions will be assigned to an existing prototype if all indicators are similar enough'''
    pass

# TODO:
#http://stackoverflow.com/questions/3288595/multiprocessing-using-pool-map-on-a-function-defined-in-a-class
#http://www.rueckstiess.net/research/snippets/show/ca1d7d90
def calculateIndicatorPipe(pairs, predParam, timeHorizon=75,collisionDistanceThreshold=1.8):  
    collisionPoints, crossingZones = prediction.computeCrossingsCollisions(pairs.roadUser1, pairs.roadUser2, predParam, collisionDistanceThreshold, timeHorizon)      
    #print pairs.num    
    # Ignore empty collision points
    empty = 1
    for i in collisionPoints:
        if(collisionPoints[i] != []):
            empty = 0
    if(empty == 1):
        pairs.hasCP = 0
    else:
        pairs.hasCP = 1
    pairs.CP = collisionPoints
    
    # Ignore empty crossing zones
    empty = 1
    for i in crossingZones:
        if(crossingZones[i] != []):
            empty = 0
    if(empty == 1):
        pairs.hasCZ = 0
    else:
        pairs.hasCZ = 1
    pairs.CZ = crossingZones
    return pairs

def calculateIndicatorPipe_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return calculateIndicatorPipe(*a_b)

class VehPairs():
    '''Create a veh-pairs object from objects list'''
    def __init__(self,objects):
        self.pairs = createInteractions(objects)
        self.interactionCount = 0
        self.CPcount = 0
        self.CZcount = 0
    
    # Process indicator calculation with support for multi-threading
    def calculateIndicators(self,predParam,threads=1,timeHorizon=75,collisionDistanceThreshold=1.8):       
        if(threads > 1):
            pool = multiprocessing.Pool(threads)
            self.pairs = pool.map(calculateIndicatorPipe_star, itertools.izip(self.pairs, itertools.repeat(predParam)))
            pool.close()
        else:
            #prog = Tools.ProgressBar(0, len(self.pairs), 77) #Removed in traffic-intelligenc port
            for j in xrange(len(self.pairs)):
                #prog.updateAmount(j) #Removed in traffic-intelligenc port
                collisionPoints, crossingZones = prediction.computeCrossingsCollisions(self.pairs[j].roadUser1, self.pairs[j].roadUser2, predParam, collisionDistanceThreshold, timeHorizon)      
                
                # Ignore empty collision points
                empty = 1
                for i in collisionPoints:
                    if(collisionPoints[i] != []):
                        empty = 0
                if(empty == 1):
                    self.pairs[j].hasCP = 0
                else:
                    self.pairs[j].hasCP = 1
                self.pairs[j].CP = collisionPoints
                
                # Ignore empty crossing zones
                empty = 1
                for i in crossingZones:
                    if(crossingZones[i] != []):
                        empty = 0
                if(empty == 1):
                    self.pairs[j].hasCZ = 0
                else:
                    self.pairs[j].hasCZ = 1
                self.pairs[j].CZ = crossingZones       
                
        for j in self.pairs:
            self.interactionCount = self.interactionCount + len(j.CP)
        self.CPcount = len(self.getCPlist())
        self.Czcount = len(self.getCZlist())
    
    
    def getPairsWCP(self):
        lists = []
        for j in self.pairs:
            if(j.hasCP):
                lists.append(j.num)
        return lists
        
    def getPairsWCZ(self):
        lists = []
        for j in self.pairs:
            if(j.hasCZ):
                lists.append(j.num)
        return lists
    
    def getCPlist(self,indicatorThreshold=float('Inf')):
        lists = []
        for j in self.pairs:
            if(j.hasCP):
                for k in j.CP:
                    if(j.CP[k] != [] and j.CP[k][0].indicator < indicatorThreshold):
                        lists.append([k,j.CP[k][0]])
        return lists
     
    def getCZlist(self,indicatorThreshold=float('Inf')):
        lists = []
        for j in self.pairs:
            if(j.hasCZ):
                for k in j.CZ:
                    if(j.CZ[k] != [] and j.CZ[k][0].indicator < indicatorThreshold):
                        lists.append([k,j.CZ[k][0]])
        return lists
        
    def genIndicatorHistogram(self, CPlist=False, bins=range(0,100,1)):
        if(not CPlist):
            CPlist = self.getCPlist()
        if(not CPlist):
            return False
        TTC_list = []
        for i in CPlist:
            TTC_list.append(i[1].indicator)
        histo = np.histogram(TTC_list,bins=bins)
        histo += (histo[0].astype(float)/np.sum(histo[0]),)
        return histo

class Crossing(moving.STObject):
    '''Class for the event of a street crossing

    TODO: detecter passage sur la chaussee
    identifier origines et destination (ou uniquement chaussee dans FOV)
    carac traversee
    detecter proximite veh (retirer si trop similaire simultanement
    carac interaction'''
    
    def __init__(self, roaduserNum = None, num = None, timeInterval = None):
        moving.STObject.__init__(self, num, timeInterval)
        self.roaduserNum = roaduserNum

    

if __name__ == "__main__":
    import doctest
    import unittest
    suite = doctest.DocFileSuite('tests/events.txt')
    #suite = doctest.DocTestSuite()
    unittest.TextTestRunner().run(suite)
    
