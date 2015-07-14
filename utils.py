#! /usr/bin/env python
''' Generic utilities.'''

#from numpy import *
#from pylab import *
from datetime import time, datetime
from math import sqrt

__metaclass__ = type

datetimeFormat = "%Y-%m-%d %H:%M:%S"

#########################
# Enumerations
#########################

def inverseEnumeration(l):
    'Returns the dictionary that provides for each element in the input list its index in the input list'
    result = {}
    for i,x in enumerate(l):
        result[x] = i
    return result

#########################
# Simple statistics
#########################

def sampleSize(stdev, tolerance, percentConfidence, printLatex = False):
    from scipy.stats.distributions import norm
    k = round(norm.ppf(0.5+percentConfidence/200., 0, 1)*100)/100. # 1.-(100-percentConfidence)/200.
    if printLatex:
        print('${0}^2\\frac{{{1}^2}}{{{2}^2}}$'.format(k, stdev, tolerance))
    return (k*stdev/tolerance)**2

def confidenceInterval(mean, stdev, nSamples, percentConfidence, trueStd = True, printLatex = False):
    '''if trueStd, use normal distribution, otherwise, Student

    Use otherwise t.interval or norm.interval
    ex: norm.interval(0.95, loc = 0., scale = 2.3/sqrt(11))
    t.interval(0.95, 10, loc=1.2, scale = 2.3/sqrt(nSamples))
    loc is mean, scale is sigma/sqrt(n) (for Student, 10 is df)'''
    from math import sqrt
    from scipy.stats.distributions import norm, t
    if trueStd:
        k = round(norm.ppf(0.5+percentConfidence/200., 0, 1)*100)/100. # 1.-(100-percentConfidence)/200.
    else: # use Student
         k = round(t.ppf(0.5+percentConfidence/200., nSamples-1)*100)/100.
    e = k*stdev/sqrt(nSamples)
    if printLatex:
        print('${0} \pm {1}\\frac{{{2}}}{{\sqrt{{{3}}}}}$'.format(mean, k, stdev, nSamples))
    return mean-e, mean+e

def computeChi2(expected, observed):
    '''Returns the Chi2 statistics'''
    result = 0.
    for e, o in zip(expected, observed):
        result += ((e-o)*(e-o))/e
    return result

class EmpiricalDistribution:
    def nSamples(self):
        return sum(self.counts)

def cumulativeDensityFunction(sample, normalized = False):
    '''Returns the cumulative density function of the sample of a random variable'''
    from numpy import arange, cumsum
    xaxis = sorted(sample)
    counts = arange(1,len(sample)+1) # dtype = float
    if normalized:
        counts /= float(len(sample))
    return xaxis, counts

class EmpiricalDiscreteDistribution(EmpiricalDistribution):
    '''Class to represent a sample of a distribution for a discrete random variable
    '''
    from numpy.core.fromnumeric import sum

    def __init__(self, categories, counts):
        self.categories = categories
        self.counts = counts

    def mean(self):
        result = [float(x*y) for x,y in zip(self.categories, self.counts)]
        return sum(result)/self.nSamples()

    def var(self, mean = None):
        if not mean:
            m = self.mean()
        else:
            m = mean
        result = 0.
        squares = [float((x-m)*(x-m)*y) for x,y in zip(self.categories, self.counts)]
        return sum(squares)/(self.nSamples()-1)

    def referenceCounts(self, probability):
        '''probability is a function that returns the probability of the random variable for the category values'''
        refProba = [probability(c) for c in self.categories]
        refProba[-1] = 1-sum(refProba[:-1])
        refCounts = [r*self.nSamples() for r in refProba]
        return refCounts, refProba

class EmpiricalContinuousDistribution(EmpiricalDistribution):
    '''Class to represent a sample of a distribution for a continuous random variable
    with the number of observations for each interval
    intervals (categories variable) are defined by their left limits, the last one being the right limit
    categories contain therefore one more element than the counts'''
    def __init__(self, categories, counts):
        # todo add samples for initialization and everything to None? (or setSamples?)
        self.categories = categories
        self.counts = counts

    def mean(self):
        result = 0.
        for i in range(len(self.counts)-1):
            result += self.counts[i]*(self.categories[i]+self.categories[i+1])/2
        return result/self.nSamples()

    def var(self, mean = None):
        if not mean:
            m = self.mean()
        else:
            m = mean
        result = 0.
        for i in range(len(self.counts)-1):
            mid = (self.categories[i]+self.categories[i+1])/2
            result += self.counts[i]*(mid - m)*(mid - m)
        return result/(self.nSamples()-1)

    def referenceCounts(self, cdf):
        '''cdf is a cumulative distribution function
        returning the probability of the variable being less that x'''
        # refCumulativeCounts = [0]#[cdf(self.categories[0][0])]
#         for inter in self.categories:
#             refCumulativeCounts.append(cdf(inter[1]))
        refCumulativeCounts = [cdf(x) for x in self.categories[1:-1]]

        refProba = [refCumulativeCounts[0]]
        for i in xrange(1,len(refCumulativeCounts)):
            refProba.append(refCumulativeCounts[i]-refCumulativeCounts[i-1])
        refProba.append(1-refCumulativeCounts[-1])
        refCounts = [p*self.nSamples() for p in refProba]
        
        return refCounts, refProba

    def printReferenceCounts(self, refCounts=None):
        if refCounts:
            ref = refCounts
        else:
            ref = self.referenceCounts
        for i in xrange(len(ref[0])):
            print('{0}-{1} & {2:0.3} & {3:0.3} \\\\'.format(self.categories[i],self.categories[i+1],ref[1][i], ref[0][i]))


#########################
# maths section
#########################

# def kernelSmoothing(sampleX, X, Y, weightFunc, halfwidth):
#     '''Returns a smoothed weighted version of Y at the predefined values of sampleX
#     Sum_x weight(sample_x,x) * y(x)'''
#     from numpy import zeros, array
#     smoothed = zeros(len(sampleX))
#     for i,x in enumerate(sampleX):
#         weights = array([weightFunc(x,xx, halfwidth) for xx in X])
#         if sum(weights)>0:
#             smoothed[i] = sum(weights*Y)/sum(weights)
#         else:
#             smoothed[i] = 0
#     return smoothed

def kernelSmoothing(x, X, Y, weightFunc, halfwidth):
    '''Returns the smoothed estimate of (X,Y) at x
    Sum_x weight(sample_x,x) * y(x)'''
    from numpy import zeros, array
    weights = array([weightFunc(x,observedx, halfwidth) for observedx in X])
    if sum(weights)>0:
        return sum(weights*Y)/sum(weights)
    else:
        return 0

def uniform(center, x, halfwidth):
    if abs(center-x)<halfwidth:
        return 1.
    else:
        return 0.

def gaussian(center, x, halfwidth):
    from numpy import exp
    return exp(-((center-x)/halfwidth)**2/2)

def epanechnikov(center, x, halfwidth):
    diff = abs(center-x)
    if diff<halfwidth:
        return 1.-(diff/halfwidth)**2
    else:
        return 0.
    
def triangular(center, x, halfwidth):
    diff = abs(center-x)
    if diff<halfwidth:
        return 1.-abs(diff/halfwidth)
    else:
        return 0.

def medianSmoothing(x, X, Y, halfwidth):
    '''Returns the media of Y's corresponding to X's in the interval [x-halfwidth, x+halfwidth]'''
    from numpy import median
    return median([y for observedx, y in zip(X,Y) if abs(x-observedx)<halfwidth])

def argmaxDict(d):
    return max(d, key=d.get)

def framesToTime(nFrames, frameRate, initialTime = time()):
    '''returns a datetime.time for the time in hour, minutes and seconds
    initialTime is a datetime.time'''
    from math import floor
    seconds = int(floor(float(nFrames)/float(frameRate))+initialTime.hour*3600+initialTime.minute*60+initialTime.second)
    h = int(floor(seconds/3600.))
    seconds = seconds - h*3600
    m = int(floor(seconds/60))
    seconds = seconds - m*60
    return time(h, m, seconds)

def timeToFrames(t, frameRate):
    return frameRate*(t.hour*3600+t.minute*60+t.second)

def sortXY(X,Y):
    'returns the sorted (x, Y(x)) sorted on X'
    D = {}
    for x, y in zip(X,Y):
        D[x]=y
    xsorted = sorted(D.keys())
    return xsorted, [D[x] for x in xsorted]

def ceilDecimals(v, nDecimals):
    '''Rounds the number at the nth decimal
    eg 1.23 at 0 decimal is 2, at 1 decimal is 1.3'''
    from math import ceil,pow
    tens = pow(10,nDecimals)
    return ceil(v*tens)/tens

def inBetween(bound1, bound2, x):
    return bound1 <= x <= bound2 or bound2 <= x <= bound1

def pointDistanceL2(x1,y1,x2,y2):
    ''' Compute point-to-point distance (L2 norm, ie Euclidean distance)'''
    return sqrt((x2-x1)**2+(y2-y1)**2)

def crossProduct(l1, l2):
    return l1[0]*l2[1]-l1[1]*l2[0]

def cat_mvgavg(cat_list, halfWidth):
    ''' Return a list of categories/values smoothed according to a window. 
        halfWidth is the search radius on either side'''
    from copy import deepcopy
    smoothed = deepcopy(cat_list)
    for point in range(len(cat_list)):
        lower_bound_check = max(0,point-halfWidth)
        upper_bound_check = min(len(cat_list)-1,point+halfWidth+1)
        window_values = cat_list[lower_bound_check:upper_bound_check]
        smoothed[point] = max(set(window_values), key=window_values.count)
    return smoothed

def filterMovingWindow(inputSignal, halfWidth):
    '''Returns an array obtained after the smoothing of the input by a moving average
    The first and last points are copied from the original.'''
    from numpy import ones,convolve,array
    width = float(halfWidth*2+1)
    win = ones(width,'d')
    result = convolve(win/width,array(inputSignal),'same')
    result[:halfWidth] = inputSignal[:halfWidth]
    result[-halfWidth:] = inputSignal[-halfWidth:]
    return result

def linearRegression(x, y, deg = 1, plotData = False):
    '''returns the least square estimation of the linear regression of y = ax+b
    as well as the plot'''
    from numpy.lib.polynomial import polyfit
    from matplotlib.pyplot import plot
    from numpy.core.multiarray import arange
    coef = polyfit(x, y, deg)
    if plotData:
        def poly(x):
            result = 0
            for i in range(len(coef)):
                result += coef[i]*x**(len(coef)-i-1)
            return result
        plot(x, y, 'x')
        xx = arange(min(x), max(x),(max(x)-min(x))/1000)
        plot(xx, [poly(z) for z in xx])
    return coef

#########################
# iterable section
#########################

def mostCommon(L):
    '''Returns the most frequent element in a iterable

    taken from http://stackoverflow.com/questions/1518522/python-most-common-element-in-a-list'''
    from itertools import groupby
    from operator import itemgetter
    # get an iterable of (item, iterable) pairs
    SL = sorted((x, i) for i, x in enumerate(L))
    # print 'SL:', SL
    groups = groupby(SL, key=itemgetter(0))
    # auxiliary function to get "quality" for an item
    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(L)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
            # print 'item %r, count %r, minind %r' % (item, count, min_index)
        return count, -min_index
    # pick the highest-count/earliest item
    return max(groups, key=_auxfun)[0]

#########################
# sequence section
#########################

class LCSS:
    '''Class that keeps the LCSS parameters
    and puts together the various computations'''
    def __init__(self, similarityFunc, delta = float('inf'), aligned = False, lengthFunc = min):
        self.similarityFunc = similarityFunc
        self.aligned = aligned
        self.delta = delta
        self.lengthFunc = lengthFunc
        self.subSequenceIndices = [(0,0)]

    def similarities(self, l1, l2, jshift=0):
        from numpy import zeros, int as npint
        n1 = len(l1)
        n2 = len(l2)
        self.similarityTable = zeros((n1+1,n2+1), dtype = npint)
        for i in xrange(1,n1+1):
            for j in xrange(max(1,i-jshift-self.delta),min(n2,i-jshift+self.delta)+1):
                if self.similarityFunc(l1[i-1], l2[j-1]):
                    self.similarityTable[i,j] = self.similarityTable[i-1,j-1]+1
                else:
                    self.similarityTable[i,j] = max(self.similarityTable[i-1,j], self.similarityTable[i,j-1])

    def subSequence(self, i, j):
        '''Returns the subsequence of two sequences
        http://en.wikipedia.org/wiki/Longest_common_subsequence_problem'''
        if i == 0 or j == 0:
            return []
        elif self.similarityTable[i][j] == self.similarityTable[i][j-1]:
            return self.subSequence(i, j-1)
        elif self.similarityTable[i][j] == self.similarityTable[i-1][j]:
            return self.subSequence(i-1, j)
        else:
            return self.subSequence(i-1, j-1) + [(i-1,j-1)]

    def _compute(self, _l1, _l2, computeSubSequence = False):
        '''returns the longest common subsequence similarity
        based on the threshold on distance between two elements of lists l1, l2
        similarityFunc returns True or False whether the two points are considered similar

        if aligned, returns the best matching if using a finite delta by shifting the series alignments

        eg distance(p1, p2) < epsilon
        '''
        if len(_l2) < len(_l1): # l1 is the shortest
            l1 = _l2
            l2 = _l1
            revertIndices = True
        else:
            l1 = _l1
            l2 = _l2
            revertIndices = False
        n1 = len(l1)
        n2 = len(l2)

        if self.aligned:
            lcssValues = {}
            similarityTables = {}
            for i in xrange(-n2-self.delta+1, n1+self.delta): # interval such that [i-shift-delta, i-shift+delta] is never empty, which happens when i-shift+delta < 1 or when i-shift-delta > n2
                self.similarities(l1, l2, i)
                lcssValues[i] = self.similarityTable.max()
                similarityTables[i] = self.similarityTable
                #print self.similarityTable
            alignmentShift = argmaxDict(lcssValues) # ideally get the medium alignment shift, the one that minimizes distance
            self.similarityTable = similarityTables[alignmentShift]
        else:
            alignmentShift = 0
            self.similarities(l1, l2)

        # threshold values for the useful part of the similarity table are n2-n1-delta and n1-n2-delta
        self.similarityTable = self.similarityTable[:min(n1, n2+alignmentShift+self.delta)+1, :min(n2, n1-alignmentShift+self.delta)+1]

        if computeSubSequence:
            self.subSequenceIndices = self.subSequence(self.similarityTable.shape[0]-1, self.similarityTable.shape[1]-1)
            if revertIndices:
                self.subSequenceIndices = [(j,i) for i,j in self.subSequenceIndices]
        return self.similarityTable[-1,-1]

    def compute(self, l1, l2, computeSubSequence = False):
        '''get methods are to be shadowed in child classes '''
        return self._compute(l1, l2, computeSubSequence)

    def computeAlignment(self):
        from numpy import mean
        return mean([j-i for i,j in self.subSequenceIndices])

    def _computeNormalized(self, l1, l2, computeSubSequence = False):
        ''' compute the normalized LCSS
        ie, the LCSS divided by the min or mean of the indicator lengths (using lengthFunc)
        lengthFunc = lambda x,y:float(x,y)/2'''
        return float(self._compute(l1, l2, computeSubSequence))/self.lengthFunc(len(l1), len(l2))

    def computeNormalized(self, l1, l2, computeSubSequence = False):
        return self._computeNormalized(l1, l2, computeSubSequence)

    def _computeDistance(self, l1, l2, computeSubSequence = False):
        ''' compute the LCSS distance'''
        return 1-self._computeNormalized(l1, l2, computeSubSequence)

    def computeDistance(self, l1, l2, computeSubSequence = False):
        return self._computeDistance(l1, l2, computeSubSequence)
    
#########################
# plotting section
#########################

def plotPolygon(poly, options = ''):
    'Plots shapely polygon poly'
    from numpy.core.multiarray import array
    from matplotlib.pyplot import plot
    from shapely.geometry import Polygon

    tmp = array(poly.exterior)
    plot(tmp[:,0], tmp[:,1], options)

def stepPlot(X, firstX, lastX, initialCount = 0, increment = 1):
    '''for each value in X, increment by increment the initial count
    returns the lists that can be plotted 
    to obtain a step plot increasing by one for each value in x, from first to last value
    firstX and lastX should be respectively smaller and larger than all elements in X'''
    
    sortedX = []
    counts = [initialCount]
    for x in sorted(X):
        sortedX += [x,x]
        counts.append(counts[-1])
        counts.append(counts[-1]+increment)
    counts.append(counts[-1])
    return [firstX]+sortedX+[lastX], counts

class PlottingPropertyValues:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, i):
        return self.values[i%len(self.values)]

markers = PlottingPropertyValues(['+', '*', ',', '.', 'x', 'D', 's', 'o'])
scatterMarkers = PlottingPropertyValues(['s','o','^','>','v','<','d','p','h','8','+','x'])

linestyles = PlottingPropertyValues(['-', '--', '-.', ':'])

colors = PlottingPropertyValues('brgmyck') # 'w'

def plotIndicatorMap(indicatorMap, squareSize, masked = True, defaultValue=-1):
    from numpy import array, arange, ones, ma
    from matplotlib.pyplot import pcolor
    coords = array(indicatorMap.keys())
    minX = min(coords[:,0])
    minY = min(coords[:,1])
    X = arange(minX, max(coords[:,0])+1.1)*squareSize
    Y = arange(minY, max(coords[:,1])+1.1)*squareSize
    C = defaultValue*ones((len(Y), len(X)))
    for k,v in indicatorMap.iteritems():
        C[k[1]-minY,k[0]-minX] = v
    if masked:
        pcolor(X, Y, ma.masked_where(C==defaultValue,C))
    else:
        pcolor(X, Y, C)

#########################
# Data download
#########################

def downloadECWeather(stationID, years, months = [], outputDirectoryname = '.', english = True):
    '''Downloads monthly weather data from Environment Canada
    If month is provided (number 1 to 12), it means hourly data for the whole month
    Otherwise, means the data for each day, for the whole year

    Example: MONTREAL MCTAVISH	10761
             MONTREALPIERRE ELLIOTT TRUDEAU INTL A	5415

    To get daily data for 2010 and 2011, downloadECWeather(10761, [2010,2011], [], '/tmp')
    To get hourly data for 2009 and 2012, January, March and October, downloadECWeather(10761, [2009,2012], [1,3,10], '/tmp')'''
    import urllib2
    if english:
        language = 'e'
    else:
        language = 'f'
    if len(months) == 0:
        timeFrame = 2
        months = [1]
    else:
        timeFrame = 1

    for year in years:
        for month in months:
            url = urllib2.urlopen('http://climat.meteo.gc.ca/climateData/bulkdata_{}.html?format=csv&stationID={}&Year={}&Month={}&Day=1&timeframe={}&submit=++T%C3%A9l%C3%A9charger+%0D%0Ades+donn%C3%A9es'.format(language, stationID, year, month, timeFrame))
            data = url.read()
            outFilename = '{}/{}-{}'.format(outputDirectoryname, stationID, year)
            if timeFrame == 1:
                outFilename += '-{}-hourly'.format(month)
            else:
                outFilename += '-daily'
            outFilename += '.csv'
            out = open(outFilename, 'w')
            out.write(data)
            out.close()

#########################
# File I/O
#########################

def removeExtension(filename, delimiter = '.'):
    '''Returns the filename minus the extension (all characters after last .)'''
    i = filename.rfind(delimiter)
    if i>0:
        return filename[:i]
    else:
        return filename

def cleanFilename(s):
    'cleans filenames obtained when contatenating figure characteristics'
    return s.replace(' ','-').replace('.','').replace('/','-')

def listfiles(dirname, extension, remove = False):
    '''Returns the list of files with the extension in the directory dirname
    If remove is True, the filenames are stripped from the extension'''
    from os import listdir
    tmp = [f for f in listdir(dirname) if f.endswith(extension)]
    tmp.sort()
    if remove:
        return [removeExtension(f, extension) for f in tmp]
    else:
        return tmp

def mkdir(dirname):
    'Creates a directory if it does not exist'
    import os
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    else:
        print(dirname+' already exists')

def removeFile(filename):
    '''Deletes the file while avoiding raising an error 
    if the file does not exist'''
    import os
    if (os.path.exists(filename)):
        os.remove(filename)
    else:
        print(filename+' does not exist')

def line2Floats(l, separator=' '):
    '''Returns the list of floats corresponding to the string'''
    return [float(x) for x in l.split(separator)]

def line2Ints(l, separator=' '):
    '''Returns the list of ints corresponding to the string'''
    return [int(x) for x in l.split(separator)]

#########################
# CLI utils
#########################

def parseCLIOptions(helpMessage, options, cliArgs, optionalOptions=[]):
    ''' Simple function to handle similar argument parsing
    Returns the dictionary of options and their values

    * cliArgs are most likely directly sys.argv 
    (only the elements after the first one are considered)
    
    * options should be a list of strings for getopt options, 
    eg ['frame=','correspondences=','video=']
    A value must be provided for each option, or the program quits'''
    import sys, getopt
    from numpy.core.fromnumeric import all
    optionValues, args = getopt.getopt(cliArgs[1:], 'h', ['help']+options+optionalOptions)
    optionValues = dict(optionValues)

    if '--help' in optionValues.keys() or '-h' in optionValues.keys():
        print(helpMessage+
              '\n - Compulsory options: '+' '.join([opt.replace('=','') for opt in options])+
              '\n - Non-compulsory options: '+' '.join([opt.replace('=','') for opt in optionalOptions]))
        sys.exit()

    missingArgument = [('--'+opt.replace('=','') in optionValues.keys()) for opt in options]
    if not all(missingArgument):
        print('Missing argument')
        print(optionValues)
        sys.exit()

    return optionValues


#########################
# Profiling
#########################

def analyzeProfile(profileFilename, stripDirs = True):
    '''Analyze the file produced by cProfile 

    obtained by for example: 
    - call in script (for main() function in script)
    import cProfile, os
    cProfile.run('main()', os.path.join(os.getcwd(),'main.profile'))

    - or on the command line:
    python -m cProfile [-o profile.bin] [-s sort] scriptfile [arg]'''
    import pstats, os
    p = pstats.Stats(os.path.join(os.pardir, profileFilename))
    if stripDirs:
        p.strip_dirs()
    p.sort_stats('time')
    p.print_stats(.2)
    #p.sort_stats('time')
    # p.print_callees(.1, 'int_prediction.py:')
    return p

#########################
# running tests
#########################

if __name__ == "__main__":
    import doctest
    import unittest
    suite = doctest.DocFileSuite('tests/utils.txt')
    #suite = doctest.DocTestSuite()
    unittest.TextTestRunner().run(suite)
    #doctest.testmod()
    #doctest.testfile("example.txt")
