import matplotlib.pyplot as plt
import numpy as np
from transmissionGraphs import readSampleData, readMorphologyData, readTransmission,readCyclicVoltAmperometry, alignData, plotData
#from stressMeasurement import plotData
import sys

if __name__ == "__main__":
    sampleRun = sys.argv[1]
    displayPlot = True
    #initDir = "C://Users/Cavan Day-Lewis/Google Drive/M Sci Project - Smart Windows 2018/Data"
    #initDir = "E://"
    initDir = "C://Users/Cavan Day-Lewis/OneDrive - University of Bristol/Important Stuff/Cavan Day-Lewis/Bristol University/Physics Year 4/Year 4 Project/"
    
    
def cyclicVoltametry(current, voltage, transmission, timeInt, sampleRun, displayTimePlot = False, displayVoltagePlot = False, saveFig = False):
    #plot position against time
    cycles = current.shape[1]
    nRows = current.shape[0]
    
    currentR = current.ravel(order = 'F')
    voltageR = voltage.ravel(order = 'F')
    timeCurr = []
    VPS = 0.06
    SPx = 2/(nRows*VPS)
    for k in range(cycles):
        timeTemp = [(k*nRows+i)*SPx for i in range(nRows)]
        timeCurr = timeCurr+timeTemp
    transmission, timeInt = alignData(transmission, timeInt, timeCurr, threshold = 0.97)

    if displayTimePlot == True:
        plotData(transmission, timeInt, currentR, timeCurr, sampleRun, voltage = voltageR, saveFig = saveFig)
        
    if displayVoltagePlot == True:
        nPoints = next(x[0] for x in enumerate(timeInt) if x[1]>SPx*2000*cycles)
        voltageInt = [0.0]*nPoints
        i=0
        n=0
        for t in timeInt:
            n = np.floor(t*VPS)
            direction = np.exp(1j*(n+1)*np.pi).real
            if i < nPoints:
                voltageInt[i] = (VPS*t-(0.5+n))*direction
            i+=1
        voltageInt = np.array(voltageInt)
        voltageIntReshape = [[]]*cycles
        transmissionReshape = [[]]*cycles
        for n in range(cycles):
            start = next(x[0] for x in enumerate(timeInt) if x[1]>SPx*2000*n)
            end = next(x[0] for x in enumerate(timeInt) if x[1]>SPx*2000*(n+1))
            #print(start, end, len(voltageIntReshape), n)
            voltageIntReshape[n] = voltageInt[start:end]
            transmissionReshape[n] = transmission[start:end]
        #voltageInt = voltageInt.reshape((int(nPoints/(cycles)),cycles))
        #transmissionReshape = np.array(transmission[:nPoints]).reshape(voltageInt.shape)
        
        plt.plot(voltage[:,0],current[:,1])
        plt.xlabel('Voltage (V)', fontsize = 30)
        plt.ylabel(r'Current ($\mu m$)', fontsize = 30)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.grid(True)
        fig = plt.gcf()
        fig.set_size_inches(16.5,9.5)
        plt.show()
        
        fig, ax1 = plt.subplots() #beigin plotting
        #for i in range(cycles):
        ax1.plot(voltage[:,0],current[:,1], 'b-')
        ax1.set_xlabel('Voltage (V)', fontsize = 30)
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel(r'Current ($\mu m$)', color='b', fontsize = 30)
        ax1.tick_params('y', colors='b', labelsize=30)
        ax1.tick_params('x', labelsize=30)

        #plt.axhline(color="black")
        #plt.axvline(color="black")

        ax2 = ax1.twinx() #create axis on the same graph
        #for i in range(cycles):
        ax2.plot(voltageIntReshape[1],transmissionReshape[1], 'g-')
        ax2.set_ylabel('Transmission (A.U.)', color='g', fontsize =30)
        #ax1.set_ylim([ax2.get_ylim()[0]*ax1.get_ylim()[1]/ax2.get_ylim()[1], ax1.get_ylim()[1]]) # change the y axis limit of the transmission plot, so that 0 intensity is the same for both current and transmission
        ax2.tick_params('y', colors='g', labelsize = 30)
        #plt.grid(True)
        fig = plt.gcf()
        fig.set_size_inches(16.5,9.5)
        if saveFig == True:
            plt.savefig("Transmission/Graphs/"+sampleRun+"_CV_auto.png")
            plt.close()
            print("Data Saved To: Transmission/Graphs/"+sampleRun+"_CV_auto.png")
        plt.show() #fig.tight_layout() 
    
if __name__ == "__main__":
    sampleDict = readSampleData()
    sampleDict = readMorphologyData(sampleDict)
    if int(sampleRun[7:7+sampleRun[7:].find("_")]) == 14:
        batchNo="Batch_3"
    elif int(sampleRun[7:7+sampleRun[7:].find("_")]) == 2 or int(sampleRun[7:7+sampleRun[7:].find("_")]) == 5:
        batchNo="Batch_2"
    else:
        batchNo="Batch_1"

    amperometryName = initDir+"Cyclic Voltametry/"+batchNo+"/"+sampleRun+"/"+sampleRun+"_CV_current.csv"  #name of amperometry file
    direc = initDir+"Cyclic Voltametry/"+batchNo+"/"+sampleRun+"/"
    
    transmission, timeInt = readTransmission(direc, sampleRun, sampleDict,sampleRun, batchNo)
    current, voltage = readCyclicVoltAmperometry(amperometryName)
    cyclicVoltametry(current, voltage, transmission, timeInt, sampleRun, displayPlot, displayPlot, False)
    
    
