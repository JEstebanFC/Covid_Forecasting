# Main idea, give the possibility of:
#    - Save in real time the results in a file
#    - Print just specify results of all the parameters that would be saved in a table format
#    - Helps in the plotting, doing a normalize version of this and using the already take values

__author = "$Author: jfarina $"

import os
import sys
import csv
import struct
import itertools
import traceback

from time import gmtime,strftime
from math import ceil,floor,trunc

import pandas as pd
import matplotlib
try:
    if matplotlib.get_backend() != 'agg':
        matplotlib.use('Agg')
except:
    pass
import pylab as pl

from Utils.Formater import Formater

def StrToNum(string):
    if string.replace('-','').isdigit():
        return int(string)
    try:
        numFloat = float(string)
    except:
        return string
    sign = 1
    if numFloat < 0:
        sign = -1
    numFloat *= sign
    exp = 0
    if trunc(numFloat) <= 1:
        while(trunc(numFloat) <= 0):
            numFloat *= 10
            exp += 1
            if exp > 11:
                break
        numFloat = sign*round(numFloat,4)*10**-exp
    else:
        while(trunc(numFloat)/10 > 0):
            numFloat /= 10
            exp += -1
        numFloat = sign*round(numFloat*10**-exp,4)
    return numFloat

class PlotTools:
    def __init__(self, labels, X, Y):
        self.fig = pl.figure()
        self.axs = {}
        self.yMax = {}
        self.yMin = {}
        self.plots = {}
        self.titles = {}
        self.xLabel = {}
        self.yLabel = {}
        self.yDelta = {}
        self.legends = {}
        
        self.multLegendSpace = {}
        self.notUsedColors = {}
        self.colors = ['c','b','g','m','y','k','r']

        if type(labels) == str:
            labels = labels.split(',')
        graphQuant = X*Y
        if len(labels) != graphQuant:
            print("Quantity error")
            return

        for (x,y),label in zip(itertools.product(range(1,X+1),range(1,Y+1)),labels):
            z = x + (y-1)*X
            self.axs[label] = [self.fig.add_subplot(Y,X,z)]  #List[0]: Normal Plot. List[1]: TwinX plot. PENDING: List[2]: TwinY Plot (maybe dict is better)
            self.plots[label] = []
            self.legends[label] = []
            self.notUsedColors[label] = self.colors[:]

    def chooseColor(self,graphic, newColor=None):
        '''
            The list is for plot, this mean that xtwin is also considerate
        '''
        if not self.notUsedColors[graphic]:
            print('In graphic {Graphic} was exceeded the maximum quantity of colors'.format(Graphic=graphic))
            self.notUsedColors[graphic] = self.colors[:]
        if not newColor:
            newColor = self.notUsedColors[graphic][0]
        if newColor not in self.notUsedColors[graphic]:
            print('In graphic {Graphic} the color {Color} is already used'.format(Graphic=graphic,Color=newColor))
        else:
            self.notUsedColors[graphic].remove(newColor)
        return newColor

    def getAx(self,graphic,xtwin=False,add=False):
        if graphic not in self.axs:
            print('Name {Graphic} not a valid plot name'.format(Graphic=graphic))
            return
        ind = 0
        if xtwin:
            ind = 1
            if add and len(self.axs[graphic]) == 1:
                ax = self.axs[graphic][0].twinx()
                self.axs[graphic].append(ax)
            if len(self.axs[graphic]) != 2:
                # print('Plot name {Graphic} has no xtwin axis define'.format(Graphic=graphic))
                return
        return self.axs[graphic][ind]

    def getRange(self,graphic,axis,xtwin=False):
        ax = self.getAx(graphic,xtwin)
        Mins = []
        Maxs = []
        for line in ax.get_lines():
            if axis.lower() == 'x':
                data = line.get_xdata()
            if axis.lower() == 'y':
                data = line.get_ydata()
            Mins.append(min(map(float,data)))
            Maxs.append(max(map(float,data)))
        Range = [float(min(Mins)),float(max(Maxs))]
        return Range

    def addCircle(self,graphic,xCenter,yCenter,R,xtwin=False):
        circ=pl.Circle((xCenter,yCenter),radius=R,fc='none',lw=1.5)
        ax = self.getAx(graphic,xtwin)
        ax.add_patch(circ)
        #With this the plot don't looks like a circle, this need to check the xLimits!
        ax.set_ylim(yCenter-1.5*R,yCenter+1.5*R)

    def addEllipse(self,graphic,xCenter,yCenter,xLarge,yLarge,xtwin=False):
        from matplotlib.patches import Ellipse
        e=Ellipse([xCenter,yCenter],width=xLarge,height=yLarge,lw=1.5)
        ax = self.getAx(graphic,xtwin)
        ax.add_artist(e)

    def addLine(self,graphic,axis,magnitude,Range=None,color=None,lineStyle=None,xtwin=False):
        color = self.chooseColor(graphic,color)
        if not lineStyle:
            lineStyle = '-'
        ax = self.getAx(graphic,xtwin)

        if axis.lower() == 'y':
            if not Range:
                print('Not possible to add Y-Line, check range used')
                return
            ax.axvline(x=magnitude,ymin=Range[0],ymax=Range[1],c=color,ls=lineStyle,lw=1.5)

        if axis.lower() == 'x':
            # axhline is not used because the Range must be in a percent of the plot
            # which is calculated later
            xData = self.getRange(graphic,'x',xtwin)
            yData = [magnitude,magnitude]
            ax.plot(xData,yData,c=color,ls=lineStyle,lw=1.5)

    def addData(self,graphic,xData,yData,xtwin=False,legends=None,color=None,lineStyle=None,marker=None):
        ax = self.getAx(graphic,xtwin,add=True)
        color = self.chooseColor(graphic, color)
        if not lineStyle:
            lineStyle = '-'
        if marker == None:
            marker = ''
        plot, = ax.plot(xData,yData,c=color,ls=lineStyle,lw=1.5,marker=marker)
        if legends:
            self.plots[graphic].append(plot)
            self.legends[graphic].append(legends)

    def getStep(self,step):
        step = abs(step)
        exp = 0
        if trunc(step) <= 1:
            while(trunc(step) <= 0):
                step *= 10
                exp += 1
                if exp > 11:
                    break
        else:
            while(trunc(step)/10 > 0):
                step /= 10
                exp += -1
        if ceil(step) > 5:
            step = 10
        elif ceil(step) > 2:
            step = 5
        elif ceil(step) > 1:
            step = 2
        else:
            step = 1
        step *= 10**-exp
        return step,exp

    def adjustFig(self):
        #Space parameters that define the plots spacing
        xSpaceLeft    = 1
        xSpaceBetween = 1
        xSpaceRight   = 1
        ySpaceDown    = 1     #Comments request more space (PENDING)
        ySpaceBetween = 1.3
        ySpaceUpper   = 1

        height = 5
        factor = 0.3
        maxPointNumber = 30
        yMaxSpaceNumber = 8.0

        plotWidth = maxPointNumber*factor

        Y,X,z = self.axs[self.axs.keys()[0]][0].get_geometry()
        z = lambda x,y: x+(Y-y)*X
        Zx = []
        for x in range(1,X+1):
            zx = []
            for y in range(1,Y+1):
                zx.append(z(x,y))
            Zx.append(zx)
        xSpaceBetweenList = [xSpaceBetween]*len(Zx)
        for label in self.axs:
            if len(self.axs[label]) > 1:
                Y,X,z = self.axs[label][0].get_geometry()
                for i,zxs in enumerate(Zx[:-1]):
                    if z in zxs:
                        xSpaceBetweenList[i+1] = 2 * xSpaceBetween
                        break
        xSize = xSpaceLeft + xSpaceRight + sum(xSpaceBetweenList[1:]) + plotWidth*X
        ySize = ySpaceDown + ySpaceUpper + ySpaceBetween*(Y-1) + height*Y

        for label in self.axs:
            for k,ax in enumerate(self.axs[label]):
                lines = ax.get_lines()
                xtwin = k == 1

                #Y-axis Step and Range
                [yMin,yMax] = self.getRange(label,'y',xtwin)
                if label in self.yMin and self.yMin[label][k] != None:
                    yMin = max(self.yMin[label][k],yMin)
                if label in self.yMax and self.yMax[label][k] != None:
                    yMax = min(self.yMax[label][k],yMax)

                yAxisStep = (yMax - yMin) / float(yMaxSpaceNumber)
                if label in self.yDelta and self.yDelta[label][k]:
                    yAxisStep = self.yDelta[label][k]
                yAxisStep,yExp = self.getStep(yAxisStep)

                new_min = floor((yMin) / yAxisStep) * yAxisStep
                if (yMin - new_min) < yAxisStep/5.0:
                    new_min = floor((yMin - yAxisStep) / yAxisStep) * yAxisStep
                

                if label not in self.multLegendSpace:
                    self.multLegendSpace[label] = 2
                new_max =  ceil((yMax + yAxisStep*self.multLegendSpace[label]) / yAxisStep) * yAxisStep
                
                yMinCalc = new_min
                yMaxCalc = new_max
                stepCalc = yAxisStep
                if label in self.yMin and self.yMin[label][k] != None:
                    yMinCalc = self.yMin[label][k]
                if label in self.yMax and self.yMax[label][k] != None:
                    yMaxCalc = self.yMax[label][k]
                
                ax.set_ylim([yMinCalc,yMaxCalc])
                
                lenTick = (yMaxCalc - yMinCalc)/stepCalc + 1
                yTicks = []
                yAux = yMinCalc
                
                while(True):
                    yTicks.append(round(yAux,yExp))
                    yAux += stepCalc
                    if yAux > yMaxCalc:
                        if yTicks[-1] < round(yMaxCalc,yExp):
                            yTicks.append(yMaxCalc)
                        break
                
                #Minimum ticks required, motivation: Small variation or constant values, two possibles solutions:
                #   Reduce Tick step: Don't like, it doesn't help if the variation expected is small
                #   Increase yMin and yMax values
                #   Here was decide the second because this should happen just when the difference between the measurement values are very small small
                minYTicks = 8
                if len(yTicks) < minYTicks:
                    while(len(yTicks) < minYTicks):
                        yTicks = [yTicks[0] - stepCalc] + yTicks
                        if len(yTicks) == minYTicks:
                            break
                        yTicks += [yTicks[-1] + stepCalc]

                ax.set_yticks(yTicks)
                ax.set_yticklabels(yTicks)

                #X-axis Parameters
                if not xtwin:
                    #X-axis step and Range
                    [xStData,xEndData] = self.getRange(label,'x',None)
                    xAxisStep = (xEndData - xStData) / float(maxPointNumber)
                    xAxisStep,xExp = self.getStep(xAxisStep)
                    prevAxis = floor(xStData/float(xAxisStep))
                    nextAxis = ceil(xEndData/float(xAxisStep))
                    xSt = xAxisStep*prevAxis
                    xEnd = xAxisStep*nextAxis
                    if (xStData - xSt) < xAxisStep/5.0:
                        xSt -= xAxisStep
                    if (xEnd - xEndData) < xAxisStep/5.0:
                        xEnd += xAxisStep

                    #Setting the minimum, maximum and ticks for X axis
                    toXtick = []
                    xAux = xSt + xAxisStep
                    while(xEnd - xAux > xAxisStep/100.0):
                        toXtick.append(xAux)
                        xAux += xAxisStep
                    ax.set_xticks(toXtick)
                    ax.set_xticklabels(toXtick,rotation=45)
                    ax.grid(True)
                    #adjustPlot, Can be done at the start of the loop?
                    y,x,z = ax.get_geometry()
                    #Calculating the number/position of the plot in the same column and row
                    Kx = (z-1)%x + 1
                    Ky = abs((z-1)/x-(y-1)) + 1
                    #x parameters
                    x1 = xSpaceLeft/xSize
                    for i,zxs in enumerate(Zx):
                        if z in zxs:
                            zx = i
                            break
                    xgap = xSpaceBetweenList[zx]/xSize
                    xn = xSpaceRight/xSize
                    width = plotWidth/xSize
                    #Calculating the height parameters
                    y1 = ySpaceDown/ySize
                    ygap = ySpaceBetween/ySize
                    yn = ySpaceUpper/ySize
                    height = (1-y1-yn-ygap*(y-1))/y
                    #Calculating the origin
                    xOrigin = x1 + (Kx - 1)*(width + xgap)
                    yOrigin = y1 + (Ky - 1)*(height + ygap)
                    #Setting new position
                    newPosition = [xOrigin,yOrigin,width,height]
                ax.set_xlim([xSt,xEnd])
                ax.set_position(newPosition)

        self.fig.set_size_inches(xSize,ySize)

    def setParameters(self):
        for graphic in self.axs:
            #Getting plots
            ax = self.getAx(graphic)
            axTwin = self.getAx(graphic,xtwin=True)
            #Setting title
            title = graphic
            if graphic in self.titles:
                title = self.titles[graphic]
            if title:
                ax.set_title(title.replace('_',' '))
            #Setting labels in the axis
            if graphic in self.xLabel:
                ax.set_xlabel(self.xLabel[graphic])
            if graphic in self.yLabel:
                ax.set_ylabel(self.yLabel[graphic][0])
                if len(self.yLabel[graphic]) > 1 and axTwin:
                    axTwin.set_ylabel(self.yLabel[graphic][1])
            
            #Setting legends
            if self.legends[graphic]:
                byColumn = 2
                self.multLegendSpace[graphic] = 2
                if len(self.plots[graphic]) == len(self.legends[graphic]):
                    if len(self.plots[graphic]) > 6:
                        byColumn = 3
                        self.multLegendSpace[graphic] = 3
                    nColumn = len(self.legends[graphic])/byColumn
                    if nColumn*byColumn < len(self.legends[graphic]):
                        nColumn += 1
                    try:
                        ax.legend(self.plots[graphic],self.legends[graphic],bbox_to_anchor=(0.0,0.90,1.0,0.102),loc=1,ncol=nColumn)
                    except Exception as ex:
                        print('Error: Not possible to added legends in %s' %graphic)
                        print(ex.__str__())
                else:
                    print('Can not add legends to graphic %s for a mismatch between the label data and plots' %graphic)
            else:
                self.multLegendSpace[graphic] = 1

    def savePlot(self,fileName):
        self.setParameters()
        self.adjustFig()
        pl.savefig(fileName)
        pl.close()


class DataTools:
    def __init__(self,testName,labels=None,path=None,time=None,**opts):
        '''
            labels can only be None when opts["load"] options is True, and when this happen all columns are load inside this class
            path: by default is the current directory
            time:
                None:  Calculate internally the time
                False: Do not use time in the name
                String to indicate manually this parameter
            opts (Following values are not default):
                Report = False: This option disable the creation of the report (default: True)
                    new = False (default): If the file exist, new data is added
                    new = True: If the file already exist, this is erase and create a new one
                Print = False: To disable the real-time print, still possible to print data using PrintData()
                width: To indicate the width for the print tool
                load = True: To not create report, just load the data from a file
        '''

        self.labels = labels
        self.nameTest = testName
        self.formater = Formater()
        
        self.time = time
        if path:
            self.path = path
        else:
            self.path = os.getcwd()
            # self.path = '/users/{USER}/'.format(USER=os.environ.get('USER'))
        self.width = []
        if 'width' in opts:
            self.width = opts['width']
        self.Print = True
        if 'Print' in opts:
            self.Print = opts['Print']
        self.Report = True
        if 'Report' in opts:
            self.Report = opts['Report']
        self.loadFile = False
        if 'load' in opts:
            self.loadFile = opts['load']
            if self.loadFile:
                self.time=False
                self.Print = False
                self.Report = False

        STE = os.environ.get('LOCATION')
        if self.time != '' and self.time != False:            #Logs Files not need this
            testName = '%s_%s' %(testName,STE)
        self.setNameTest(testName,self.time)

        if self.Report:
            print('The results will be save in: %s' %(self.path))
            newFile = False
            if 'new' in opts:
                newFile = opts['new']
            if os.path.isfile(self.rFile):
                if newFile:
                    print("[WARNING]: File name already exist. Old file was erase")
                else:
                    print("[WARNING]: File name already exist. Data will be added to the file")
            if newFile:
                f = open(self.rFile,'w')
            else:
                f = open(self.rFile,'a')
            csvFile = csv.writer(f,dialect="excel")
            csvFile.writerow(self.labels)
            f.close()

        if self.loadFile:
            f = open(self.rFile,'r')
            csvFile = csv.reader(f,dialect='excel')
            self.labels = csvFile.next()
            self.print_index  = range(len(self.labels))
            self.df = pd.DataFrame([],columns=self.labels)
            for data in csvFile:
                improveData = []
                for d in data:
                    improveData.append(StrToNum(d))
                self.addData(improveData)
            f.close()
        elif self.labels:
            self.df = pd.DataFrame([],columns=self.labels)
            self.print_index = range(len(self.labels))
        else:
            print('Labels must be indicated')
            return

    def setNameTest(self,testName,time=None):
        self.time = '_'
        if time == '' or time == False:
            self.time = ''
        elif time == None:
            self.time += str(strftime("%Y-%m-%d_%H-%M-%S",gmtime()))
        elif time:
            self.time += time
        self.testName = '%s%s' %(testName,self.time)
        self.rFile = '%s/%s.csv' %(self.path,self.testName)

    def printLabel(self,columns=None):
        '''
            There are two ways to define the columns to be printed, using this tool indicating
            the label name, or modify the self.print_index manually indicating the number index.
        '''
        if not columns:
            return
        self.print_index = []
        for c in columns:
            self.print_index.append(self.labels.index(c))
        self.print_index.sort()

    def addData(self,data):
        if len(data) != len(self.labels):
            return 'Length Error'
        
        self.df.loc[len(self.df)] = data

        if self.Report and not self.loadFile:
            f = open(self.rFile,'a')
            csvFile = csv.writer(f,dialect="excel")
            csvFile.writerow(data)
            f.close()
        
        if not self.formater._rows:
            if not self.width:
                self.width = []
                for i in self.print_index:
                    w = max(len(str(self.labels[i])),len(str(data[i])))
                    self.width.append(w)
            self.formater.set_cols_width(self.width)
            self.formater.set_auto_width(False)
            print
            print_labels = []
            for i in self.print_index:
                print_labels.append(self.labels[i])
            self.formater.set_horizontal_header(print_labels, fast_print = self.Print)

        print_data = []
        for i in self.print_index:
            print_data.append(data[i])
        self.formater.insert_row(-1, print_data, fast_print = self.Print)

    def getData(self,labels=None,query=None):
        if not labels:
            labels = self.labels
        elif type(labels) == str:
            labels = labels.split(',')
            if False in [x in self.df.columns for x in labels]:
                print('Error in DataTools-getData, not all labels requested are in the database')
                return
        data = self.df.copy()
        if query:
            data = data.query(query)
        data = data[labels]
        return data

    def dataInTable(self):
        if self.formater._rows:
            return True
        return False

    def PrintData(self):
        self.formater.print_sheet()

    '''Plot Tools'''
    def definePlots(self,labels,X=1,Y=1,title=None):
        '''
            X: Define the quantity of graphics in X-axis
            Y: Define the quantity of graphics in Y-axis
            labels: To identify the graphics. Graphic's title used this as a default value
        '''
        self.plt = PlotTools(labels,X,Y)
        if not title:
            title = self.nameTest.replace('_',' ')
        self.plt.fig.suptitle(title,fontsize=20)

    def getPlot(self,graphic,xtwin=False):
        return self.plt.getAx(graphic,xtwin)

    def dataToPlot(self,graphic,xLabel,yLabels,legends=None,xtwin=False,query=None,**opts):
        '''
            Query are used to define a x-range of data, the script take the first and the
            last x value where this condition happen and plot all data between this range.
            Opts:
                color
                lineStyle or ls
                marker
        '''
        
        if type(yLabels) == str:
            yLabels = yLabels.split(',')
        if legends and type(legends) == str:
            legends = legends.split(',')
        
        color = None
        if 'color' in opts:
            color = opts['color']
            if type(color) == str:
                color = color.split(',')

        lineStyle = None
        if 'ls' in opts:
            lineStyle = opts['ls']
        elif 'linestyle' in opts:
            lineStyle = opts['linestyle']

        marker = None
        if 'marker' in opts:
            marker = opts['marker']

        L = len(yLabels)
        if not color:
            color = [None]*L
        if len(color) != L:
            print('Mismatch between Y-Labels and Colors indicated')
            return
        if not legends:
            legends = [None]*L
        if legends and len(legends) != L:
            print('Mismatch between Y-Labels and legends indicated')
            return
        
        labels = [xLabel] + yLabels
        data = self.getData(labels,query)
        if not data.empty:
            xData = list(data[xLabel])
            for i,yLabel in enumerate(yLabels):
                yData = list(data[yLabel])
                while yData.count(''):
                    ind = yData.index('')
                    xData.pop(ind)
                    yData.pop(ind)
                self.plt.addData(graphic,xData,yData,xtwin=xtwin,legends=legends[i],color=color[i],lineStyle=lineStyle,marker=marker)
        else:
            print('No Data to add:')
            print('\txLabel: {XL}\n\tyLabels: {YLs}\n\tquery: {Q}'.format(XL=xLabel,YLs=yLabels,Q=query))
            return

    def addToPlot(self,graphic,legend=None,xtwin=False,**opts):
        #Line Options
        color = None
        if 'color' in opts:
            color = opts['color']
        lineStyle = None
        if 'ls' in opts:
            lineStyle = opts['ls']
        elif 'linestyle' in opts:
            lineStyle = opts['linestyle']
        
        if 'circle' in opts:
            #circle = [Center,Radio]
            xCenter = opts['circle'][0][0]
            yCenter = opts['circle'][0][1]
            Radio = opts['circle'][1]
            self.plt.addCircle(graphic,xCenter,yCenter,Radio,xtwin=xtwin)
            
        if 'ellipse' in opts:
            #ellipse = [Center,Large]
            xCenter = opts['ellipse'][0][0]
            yCenter = opts['ellipse'][0][1]
            xLarge = opts['ellipse'][1][0]
            yLarge = opts['ellipse'][1][1]
            self.plt.addEllipse(graphic,xCenter,yCenter,xLarge,yLarge,xtwin=xtwin)
        
        if 'line' in opts:
            #line = [Axis,Magnitude,[start,end]]
            axis = opts['line'][0]
            magnitude = opts['line'][1]
            Range = None
            if axis.lower() == 'y':
                if len(opts['line']) > 2:
                    Range = opts['line'][2]
                else:
                    Range = [0,1]
            self.plt.addLine(graphic,axis,magnitude,Range,color,lineStyle,xtwin=xtwin)
        
        if 'data' in opts:
            xData = opts['data'][0]
            yData = opts['data'][1]
            self.plt.addData(graphic,xData,yData,xtwin=xtwin,legends=legend,color=color,lineStyle=lineStyle)

    def defParam(self,graphic,**opts):
        '''
            opts:
                title  = 'Title'
                xLabel = 'Label[s]'
                yMin   = [RegularMin,xTwinMin]
                yMax   = [RegularMax,xTwinMax]
                yDelta = [RegularDelta,xTwinDelta]
                yLabel = [RegularLabel,xTwinLabel]
        '''
        if 'title' in opts:
            self.plt.titles[graphic] = opts['title']

        if 'xLabel' in opts:
            self.plt.xLabel[graphic] = opts['xLabel']

        if 'yMin' in opts:
            self.plt.yMin[graphic] = opts['yMin']
        if 'yMax' in opts:
            self.plt.yMax[graphic] = opts['yMax']
        if 'yLabel' in opts:
            self.plt.yLabel[graphic] = opts['yLabel']
        if 'yDelta' in opts:
            self.plt.yDelta[graphic] = opts['yDelta']

    def savePlot(self,extraName=''):
        if extraName:
            extraName = '_%s' %extraName
        plotName = '%s%s.png' %(self.testName,extraName)
        pFile = '%s/%s' %(self.path,plotName)
        self.plt.savePlot(pFile)
        print('Plot file: %s\n' %pFile)
        return plotName

    '''Ideas'''
    # def rePlot(self,File,labels):
        # '''
            # The idea is to take the data from a file and created a plot using the internal tools
            # Labels can be str or index, in both cases this parameter is a list
            # Create dictionary to associated the name tests with the script. Get from this eng-script the data used
            #  to plot: This is definePlots, DataTpoPlot, defParam, etc...
            # Possibility to add more data to the plots
            # Save and/or show the plot?
        # '''
        # pass


