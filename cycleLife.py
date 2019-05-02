import numpy as np
import matplotlib.pyplot as plt
from transmissionGraphs import readTransmission, readAmperometry, alignData, plotData, readSampleData, readMorphologyData, colourationEfficiency


def plotCompare(transmissionArr, timeIntArr, currentArr, timeArr, CEArr,CEArr_err, int_r_Arr,int_r_Arr_err, sampleRun):
    print("CE before:", CEArr[0],"+/-",CEArr_err[0], "CE after", CEArr[1],"+/-",CEArr_err[1])
    print("Int_r before:", int_r_Arr[0],"+/-",int_r_Arr_err[0], "Int_r after", int_r_Arr[1],"+/-", int_r_Arr_err[1])
    plt.plot(timeIntArr[0], transmissionArr[0], label= "Initial")
    plt.plot(timeIntArr[1], transmissionArr[1], label= "After 3100 cycles")
    plt.legend(fontsize = 20)
    plt.xlabel("Time (s)", fontsize=30)
    plt.ylabel("Transmission (A.U.)", fontsize=30)
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)
    #plt.tight_layout()
    plt.grid(True)
    fig = plt.gcf()
    fig.set_size_inches(16, 9.5)
    plt.show()

if __name__ == "__main__":
    sampleName = "sample_14_3_19_13"
    batchNo = "Batch_3"
    sampleList = [sampleName+"_before", sampleName+"_after"]
    displayPlot = True
    #initDir = "C://Users/Cavan Day-Lewis/Google Drive/M Sci Project - Smart Windows 2018/Data"
    #initDir = "E://"
    initDir = "C://Users/Cavan Day-Lewis/OneDrive - University of Bristol/Important Stuff/Cavan Day-Lewis/Bristol University/Physics Year 4/Year 4 Project/"
    direc = initDir+"Cycle Life/"
    
    timeArr = [[]]*2
    transmissionArr = [[]]*2
    timeIntArr = [[]]*2
    currentArr = [[]]*2
    int_r = [[]]*2
    int_r_err = [[]]*2
    nu = [0.0]*2
    nu_err = [0.0]*2
    nu_vol = [0.0]*2
    nu_vol_err = [0.0]*2
    
    sampleDict = readSampleData()
    sampleDict = readMorphologyData(sampleDict)
    #thickness = sampleDict[sampleName]["thickness"]
    #thickness = 0.642
    #thicknessErr = 0.035109353
    i=0
    for sampleRun in sampleList:
        if i == 0:
            #direc = "C://Users/Cavan Day-Lewis/OneDrive - University of Bristol/Important Stuff/Cavan Day-Lewis/Bristol University/Physics Year 4/Year 4 Project/Transmission/"+batchNo+"/"
            transmissionArr[i], timeIntArr[i] = readTransmission(direc+sampleRun+"/", sampleRun, sampleDict,sampleName, batchNo)
            currentArr[i], timeArr[i] = readAmperometry(direc+sampleRun+"/"+sampleRun+"_amperometry.csv")
        else:
            #direc = initDir+"Cycle Life/"
            transmissionArr[i], timeIntArr[i] = readTransmission(direc+sampleRun+"/", sampleRun, sampleDict,sampleName, batchNo)
            currentArr[i], timeArr[i] = readAmperometry(direc+sampleRun+"/"+sampleRun+"_amperometry.csv")
        transmissionArr[i], timeIntArr[i] = alignData(transmissionArr[i], timeIntArr[i], timeArr[i])
        #CEArr[i], area, int_r_Arr[i], chargeAvg, thickness, nu_vol, CEArr_err[i], nu_vol_err, int_r_Arr_err[i]  = colourationEfficiency(transmissionArr[i], timeIntArr[i], currentArr[i], timeArr[i], sampleDict, sampleName)
        nu[i], area, int_r[i], chargeAvg, charge_err, thickness, nu_vol[i], nu_err[i], nu_vol_err[i], int_r_err[i] = colourationEfficiency(transmissionArr[i], timeIntArr[i], currentArr[i], timeArr[i], sampleDict, sampleName)
        
        #CEArr[i] = int_r_Arr[i]*area*thickness*10**-4/chargeAvg
        i+=1
    if displayPlot == True:
        plotCompare(transmissionArr, timeIntArr, currentArr, timeArr, nu, nu_err, int_r, int_r_err, sampleRun)
        #for i in range(len(sampleList)):
        #    plotData(transmissionArr[i], timeIntArr[i], currentArr[i], timeArr[i], sampleRun[i])