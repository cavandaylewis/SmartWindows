import numpy as np
import matplotlib.pyplot as plt
from transmissionGraphs import readSampleData, readTransmission

if __name__ == "__main__":
    #sampleRun = sys.argv[1]
    sampleRun = "sample_14_3_19_13"
    displayPlot = True
    #initDir = "C://Users/Cavan Day-Lewis/Google Drive/M Sci Project - Smart Windows 2018/Data"
    #initDir = "E://"
    initDir = "C://Users/Cavan Day-Lewis/OneDrive - University of Bristol/Important Stuff/Cavan Day-Lewis/Bristol University/Physics Year 4/Year 4 Project/"

    
def plotData(transmission, timeInt, sampleRun):
    #alter data as there is weird kink
    correction = 0.0956-0.0301
    timeHrs = [0.0]*(len(timeInt))
    for i in range(len(transmission)):
        if timeInt[i] >10870:
            transmission[i] = transmission[i]-correction
        timeHrs[i] = timeInt[i]/3600.0
    plt.plot(timeHrs, transmission)
    plt.xlabel("Time (Hrs)", fontsize = 30)
    plt.ylabel("Transmission (A.U.)", fontsize = 30)
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)
    plt.grid(True)
    fig = plt.gcf()
    fig.set_size_inches(16, 9.5)
    plt.savefig("Transmission/Graphs/sample_14_3_19_13_memory_1.png")
    plt.show()

if __name__ == "__main__":
    sampleDict = readSampleData()
    if int(sampleRun[7:7+sampleRun[7:].find("_")]) == 14:
        batchNo="Batch_3"
    elif int(sampleRun[7:7+sampleRun[7:].find("_")]) == 2 or int(sampleRun[7:7+sampleRun[7:].find("_")]) == 5:
        batchNo="Batch_2"
    else:
        batchNo="Batch_1"

    direc = initDir+"Transmission Lifetime/"
    transmission, timeInt = readTransmission(direc, sampleRun, sampleDict,sampleRun, batchNo, fileRestrict = None)
    if displayPlot == True:
        plotData(transmission, timeInt, sampleRun)