import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime
import time
import sys
from tqdm import tqdm
import pandas as pd

if __name__ == "__main__":
    sampleRun = sys.argv[1]
    initDir = "C://Users/Cavan Day-Lewis/OneDrive - University of Bristol/Important Stuff/Cavan Day-Lewis/Bristol University/Physics Year 4/Year 4 Project/"
    #initDir = ""
    displayPlot = False

def readSampleData():
    with open("Solution and sample/Record sheet.csv") as f: #read in the record sheet with information about all samples
        lines = f.readlines()
    f.close

    #create a dictionary which contains all information about the samples
    sampleDict = {}
    headers = lines[0].rstrip().split(",")
    vals = np.ndarray((len(lines),lines[0].count(",")+1), dtype="object_")
    nameIndex = headers.index("sample_name")

    for i in range(len(lines)-1):
        vals[i,:] = lines[i+1].rstrip().split(",")
        sampleDict[vals[i,nameIndex]] = {}
        j = 0
        for h in headers:
            if h != "sample_name":
                sampleDict[vals[i,nameIndex]][h] = vals[i,j]
            j+=1
    #print(sampleDict[sampleName]['date'])
    return sampleDict
    
def readMorphologyData(sampleDict):
    df = pd.read_excel (r'Profileometer\Thickness and Roughness.xlsx') #(use "r" before the path string to address special character, such as '\'). Don't forget to put the file name at the end of the path + '.xlsx'
    data = df.to_numpy()
    i=1
    for sampleName in data[1:,1]:
        sampleDict[sampleName]["thickness"] = data[i,6]
        sampleDict[sampleName]["thickness_err"] = data[i,7]
        sampleDict[sampleName]["deposition_time"] = data[i,2]
        i+=1
    return sampleDict

def readTransmission(direc, sampleRun, sampleDict, sampleName, batchNo, fileRestrict = 1000, plotSpectra = False):
    transmission = []
    files = [q for q in os.listdir(direc) if q.find(sampleRun)!=-1 and q.endswith(".csv")] #finds all of the files in the folder which have the samplename and are .csv files
    files = [q for q in files if q.find("amperometry")==-1 and q.find("current")==-1] #removes the amperometry file from the list
    print("%d files found" % (len(files)))
    
    if fileRestrict is not None:
        if len(files) > fileRestrict:
            freq = int(np.round(len(files)/fileRestrict))
            files = files[0::freq]
    print("Of which %d files will be used" % (len(files)))
    
    if len(files) == 0: #exit the code if no files are found
        sys.exit()
    j = 0

    if len(files) == 1: #if there is only one file then it is likely that fast sequential scanning was used.
        sequentialType = "Fast"
    else:
        sequentialType = "Timed"
        
    source = sampleDict[sampleName]['transmission_source'] #"laser" #here we define what the type of light source used was, laser, white, unknown

    if sequentialType == "Timed":
        timeInt = [0.0]*len(files)
        #for f in files: 
        for k in tqdm(range(len(files))):
            f = files[k]
            with open(direc+f) as fi: # Opens file
                lines = fi.readlines() #reads file and adds each line to a list
            fi.close()
            #timeStamp = f[-12:-4]
            
            if batchNo == "Batch_3": #do batch 3 analysis
                try:
                    longTime = time.strptime(f[-16:-8],'%H_%M_%S')
                    seconds = datetime.timedelta(hours=longTime.tm_hour,minutes=longTime.tm_min,seconds=longTime.tm_sec).total_seconds()
                    milliseconds = float(f[-7:-4])/1000
                    if j == 0:
                        dayInit = int(lines[2].rstrip()[-2:])
                except ValueError:
                    #try and get time from within file:
                    stringTime = lines[3].rstrip()[6:-2]
                    day = int(lines[2].rstrip()[-2:])
                    if len(stringTime) == 5:
                        stringTime = "0"+stringTime
                    longTime = time.strptime(stringTime,'%H%M%S')
                    if j == 0:
                        dayInit = int(lines[2].rstrip()[-2:])
                    seconds = datetime.timedelta(hours=longTime.tm_hour,minutes=longTime.tm_min,seconds=longTime.tm_sec).total_seconds()
                    seconds += (day-dayInit)*24*60*60
                    milliseconds = float(lines[3][-4:-2])/100
                if j == 0:
                    timeInit = seconds+milliseconds
                timeInt[j] = seconds+milliseconds-timeInit
                header = 33
                splitString = ";"
            else: #do batch 1 analysis
                longTime = time.strptime(f[-18:-9],'%Hh%Mm%Ss') #take the time of data collection from the file name
                seconds = datetime.timedelta(hours=longTime.tm_hour,minutes=longTime.tm_min,seconds=longTime.tm_sec).total_seconds() #convert this time into seconds
                if j == 0:
                    timeInit = seconds+float(f[-9:-6])/1000 # take the miliseconds from the file name and add this onto seconds and set it as the initialisation time
                timeInt[j] = seconds+float(f[-9:-6])/1000-timeInit # find the miliseconds time and add this onto seconds and take away the initialisation time
                header = 0
                splitString = ","
                
            wavelength = [0.0]*len(lines)
            transmissionSpec = [0.0]*len(lines)
            
            for i in range(header,len(lines[header:])): # looping over all of the lines in each file
                wavelength[i], transmissionSpec[i] = lines[i].rstrip().split(splitString) #separated by commas are the two columns, add each of these to array
                wavelength[i] = float(wavelength[i]) #convert wavelength to float
                transmissionSpec[i] = float(transmissionSpec[i]) #convert transmission to float
            
            #index = transmissionSpec.index(max(transmissionSpec))
            if source == "laser":
                index = next(x[0] for x in enumerate(wavelength) if x[1] > 639.4) #if source is laser light, find the wavelength where the intensity is highest
                start = index
                end = index
                transmission.append(transmissionSpec[index]) #add this to the transmission array
            elif source == "white": #if it is white light take the average from within a certain range of wavelengths
                start = next(x[0] for x in enumerate(wavelength) if x[1] > 665.0)
                end = next(x[0] for x in enumerate(wavelength) if x[1] > 685.0)
                transmission.append(sum(transmissionSpec[start:end])/(end-start))
            else: # if for some other reason we want just a generic transmisson value, we can use this method
                index = transmissionSpec.index(max(transmissionSpec))
                start = index-10
                end = index+10
                transmission.append(sum(transmissionSpec[start:end])/(end-start))
            if plotSpectra == True and j == 86:
                wavelength_plot = [wavelength[i] for i in range(len(wavelength)) if wavelength[i]>200]
                transmissionSpec_plot = [transmissionSpec[i] for i in range(len(wavelength)) if wavelength[i]>200]
                plt.plot(wavelength_plot, transmissionSpec_plot, "b-", label="Bleached")
            if plotSpectra == True and j == 95:
                wavelength_plot = [wavelength[i] for i in range(len(wavelength)) if wavelength[i]>200]
                transmissionSpec_plot = [transmissionSpec[i] for i in range(len(wavelength)) if wavelength[i]>200]
                plt.plot(wavelength_plot, transmissionSpec_plot, "r-", label="Coloured")
            #print(transmission[index], index)
            j+=1
        if plotSpectra == True:
            plt.xlabel("Wavelength (nm)", fontsize=30)
            plt.ylabel("Intensity (A.U.)", fontsize = 30)
            plt.axvspan(665.0, 685.0, facecolor="green", alpha = 0.5)
            plt.legend(fontsize=20)
            plt.xticks(fontsize=30)
            plt.yticks(fontsize=30)
            plt.grid(True)
            fig = plt.gcf()
            fig.set_size_inches(16.5,9.5)
            plt.show()
        #if sampleRun.find("_before") != -1 or sampleRun.find("after") != -1:
        #    print("Cycle life adjustment")
        #    start, end = timeInt[0], timeInt[-1]
        #    timeInt = np.array(np.linspace(start,end,len(files)))
        #    #print(start, end, timeInt)
            
    elif sequentialType == "Fast": # if fast sequentuil recording was used
        with open(direc+files[0]) as fi:
            lines = fi.readlines() #read the file
        fi.close()
        
        numSpectra = lines[0].count(",") #count the number of commas, this equals the number of spectra
        timeInt = [0.0]*numSpectra
        transmissionSpec = np.ndarray((len(lines)-1, numSpectra))
        wavelength = np.ndarray((len(lines)-1))
        timeLong = lines[0].rstrip().split(",") #the time of each spectra is displayed at the top of each column, put this into an array
        timeInt = [float(t[:-2])*10**(-3) for t in timeLong[1:]] # convert into seconds from miliseconds and append time
        for i in range(len(lines)-1): #for all of the wavelengths
            vals = lines[i+1].rstrip().split(",")
            wavelength[i], transmissionSpec[i,:] = vals[0], vals[1:] #separate wavelength values from transmission
            wavelength[i] = float(wavelength[i]) #convert to float
            transmissionSpec[i,:] = [float(t) for t in transmissionSpec[i,:]] #convert to float
        
        for j in range(numSpectra): #for all of the spectra #average over the intensity of wavelengths within a narrow band
            start = next(x[0] for x in enumerate(wavelength) if x[1] > 665.0)
            end = next(x[0] for x in enumerate(wavelength) if x[1] > 685.0)
            transmission.append(sum(transmissionSpec[start:end,j])/(end-start))
            
    return transmission, timeInt

    
def readCyclicVoltAmperometry(amperometryName):
    with open(amperometryName) as fi: # open the amperometry file
        lines = fi.readlines()
    fi.close()
    
    headers = 2
    vals = np.ndarray((len(lines),lines[0].count(",")+1), dtype="object_")
    voltage = np.ndarray((vals.shape[0]-headers, int(vals.shape[1]/2)))
    current = np.ndarray((vals.shape[0]-headers, int(vals.shape[1]/2)))
    #nCol = lines[3].count(",")
    for i in range(headers,len(lines)): #ignoring the headers
        vals[i,:] = lines[i].rstrip().split(",") #separated by commas is the time and the current values, import these into an array
        voltage[i-headers,:] = vals[i,0::2]
        #time[i] = float(time[i]) #and convert to float
        current[i-headers,:] = vals[i,1::2]
        #for j in range(vals.shape[1]-1):
        #    current[] = float(vals[i,j+1])
    
    return current, voltage
            
def readAmperometry(amperometryName):
    with open(amperometryName) as fi: # open the amperometry file
        lines = fi.readlines()
    fi.close()

    time = [0.0]*(len(lines))
    current = [0.0]*(len(lines))
    for i in range(2,len(lines)): #ignoring the headers
        time[i], current[i] = lines[i].rstrip().split(",") #separated by commas is the time and the current values, import these into an array
        time[i] = float(time[i]) #and convert to float
        current[i] = float(current[i])

    #maxCurrent = max(current)
    #for i in range(len(current)):
    #    current[i] = current[i]/maxCurrent
    
    return current, time

def alignData(transmission, timeInt, time, threshold = 0.7):
    if min(transmission) < 0: 
        """
        if the minimum transmission goes lower than zero bring it up to zero, without changing the maximum transmission
        this makes the graphs look better and it makes aligning easier
        Mathematically it doesn't matter as intensity is arbitrary anyway
        """
        minT = min(transmission)
        frac = max(transmission)/abs(max(transmission)-minT)
        
        for i in range(len(transmission)):
            transmission[i] = (transmission[i]-minT)*frac
        

    duration = int(time[-1]) # find the duration of the amperometry
    try: #begin aliging
        startIndex = next(x[0] for x in enumerate(transmission) if x[1] > max(transmission)*threshold) #find the first value in transmission that goes above 0.7
        startTime = timeInt[startIndex] #find the time this occurs
    except:
        startTime = 0 # if we cannot find a value in transmission that goes above 0.7 then we are unable to align the data
        print("Unable to align data")

    for i in range(len(timeInt)): #align the data according to the start time
        timeInt[i] = timeInt[i]-startTime
    
    return transmission, timeInt

def plotData(transmission, timeInt, current, time, sampleRun, voltage = None, saveFig = True):
    cycles = 5
    #voltage = np.array((len(time)))
    if voltage is None:
        voltage = [0.5 for i in range(len(time))]
        for n in range(cycles):
            start1 = next(x[0] for x in enumerate(time) if x[1]>30*n)
            end1 = next(x[0] for x in enumerate(time) if x[1]>30*n+10)
            start2 = end1+1
            if n == cycles-1:
                end2 = len(time)
            else:
                end2 = next(x[0] for x in enumerate(time) if x[1]>30*(n+1))
            voltage[start1:end1] = [0.5 for i in range(end1-start1)] 
            voltage[start2:end2] = [-0.5 for i in range(end2-start2)]
    
    #plt.subplots(3,1,sharex=True)
    ax1 = plt.subplot(311)
    plt.plot(time, voltage, 'r-')
    plt.ylabel('Voltage (V)', color='r', fontsize=20)
    plt.yticks(color='r', fontsize=20)
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    ax2 = plt.subplot(312, sharex = ax1)
    #plt.plot(time, current, 'g-')
    plt.plot(time, np.array(current)/1000, 'g-')
    #plt.ylabel(r'Current ($\mu A$)', color='g', fontsize=20)
    plt.ylabel(r'Current (mA)', color='g', fontsize=20)
    plt.yticks(color='g', fontsize=20)
    ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    ax3 = plt.subplot(313, sharex = ax1)
    plt.plot(timeInt, transmission, 'b-')
    plt.ylabel('Transmission', color='b', fontsize=20)
    plt.xlabel('Time (s)', fontsize = 30)
    plt.xlim(time[0]-1,time[-1]*1.05)
    plt.xticks(fontsize = 30)
    plt.yticks(color='b', fontsize = 20)
    
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    
    #plt.show()
    plt.tight_layout()
    if saveFig:
        plt.savefig("Transmission/Graphs/"+sampleRun+"_auto.png")
        plt.close()
    else:
        plt.show()

    #fig, ax1 = plt.subplots() #beigin plotting
    #ax1.plot(timeInt, transmission, 'b-') #plot the transmission
    #ax1.set_xlabel('Time (s)')
    # Make the y-axis label, ticks and tick labels match the line color.
    #ax1.set_ylabel('Transmission', color='b')
    #ax1.tick_params('y', colors='b')
    
    #plt.axhline(color="black")
    #plt.axvline(color="black")

    #ax2 = ax1.twinx() #create axis on the same graph
    #ax2.plot(time, current, 'g-') #plot the current
    #ax2.set_ylabel('Current (mA)', color='g')
    #ax1.set_ylim([ax2.get_ylim()[0]*ax1.get_ylim()[1]/ax2.get_ylim()[1], ax1.get_ylim()[1]]) # change the y axis limit of the transmission plot, so that 0 intensity is the same for both current and transmission
    #ax2.tick_params('y', colors='g')

    #plt.show() #fig.tight_layout() 

def colourationEfficiency(transmission, timeInt, current, time, sampleDict, sampleName):
    cycles = 5 #define how many cycles of the multistep amperometry we go through, we assume each cycle is 30 seconds with e1 = 10s and e2 = 20s

    length=float(sampleDict[sampleName]['length'])
    width=float(sampleDict[sampleName]['width'])
    area=length*width/100 ##  in cm^2

    max_intensity=[0.0]*cycles
    min_intensity=[0.0]*cycles
    total_charge=[0.0]*cycles
    total_charge_2=[0.0]*cycles
    total_charge_3=[0.0]*cycles
    total_charge_4=[0.0]*cycles
    timeSpacing = time[-1]-time[-2] #in s #here we find the time spacing  between amperometry data points

    current = np.array(current)
    for n in range(cycles): # for the cycles
        startCurr = int((n*30)//timeSpacing)+3 #we start the charge calculations
        switchCurr = int((n*30+10)//timeSpacing)+3
        endCurr = int(((n+1)*30)//timeSpacing)+3
        if endCurr > len(current):
            endCurr = len(current)
        total_charge[n] = np.trapz(current[startCurr:endCurr], dx = timeSpacing)/1000000 # total area
        total_charge_2[n] = np.trapz(current[switchCurr:endCurr], dx = timeSpacing)/1000000
        total_charge_3[n] = np.trapz(current[startCurr:switchCurr-5], dx = timeSpacing)/1000000
        total_charge_4[n] = abs(total_charge_2[n])+abs(total_charge_3[n])
        
        #total_charge[n] = sum(current[startCurr:endCurr]/1000000)*timeSpacing #total charge is the sum of the current graphs #we convert from micro amps to amps

        start = next(x[0] for x in enumerate(timeInt) if x[1]>30*n+3)
        switch1 = next(x[0] for x in enumerate(timeInt) if x[1]>30*n+10)
        switch2 = next(x[0] for x in enumerate(timeInt) if x[1]>30*n+20)
        end = next(x[0] for x in enumerate(timeInt) if x[1]>30*(n+1))
        max_intensity[n] = np.average(transmission[start:switch1])
        min_intensity[n] = np.average(transmission[switch2:end])
        #print(max_intensity[n], min_intensity[n], max(transmission[start:end]), min(transmission[start:end]))
        #max_intensity[n] = max(transmission[start:end])
        #min_intensity[n] = min(transmission[start:end])

    total_charge = np.array(total_charge)
    charge = abs(total_charge)
    max_intensity = np.array(max_intensity)
    min_intensity = np.array(min_intensity)
    
    #max_intensity = np.ones(max_intensity.shape)

    #print(max_intensity, min_intensity, charge)

    maxAvg = np.mean(max_intensity)
    minAvg = np.mean(min_intensity)
    chargeAvg = np.mean(charge)
    
    chargeAvg_2 = np.mean(abs(np.array(total_charge_2)))
    charge_err = np.std(abs(np.array(total_charge_2)))
    chargeAvg_3 = np.mean(abs(np.array(total_charge_3)))
    chargeAvg_4 = np.mean(abs(np.array(total_charge_4)))
    
    int_r = np.log(max_intensity/min_intensity)
    int_r_err = np.std(int_r)
    int_r = np.mean(int_r)

    #print("\n \n max avg:", MaxAvg," \n min avg:",MinAvg)
    #print(np.log(maxAvg/minAvg), chargeAvg)

    #print(np.log(max_intensity/min_intensity), charge)

    nu = area*np.log(max_intensity/min_intensity)/charge
    nu_2 = area*np.log(max_intensity/min_intensity)/abs(np.array(total_charge_2))
    nu_3 = area*np.log(max_intensity/min_intensity)/abs(np.array(total_charge_3))
    nu_4 = area*np.log(max_intensity/min_intensity)/abs(np.array(total_charge_4))
    #print(nu)
    #print("Colouration Efficiency=", np.mean(nu),"cm^2C^-1")
    thickness = sampleDict[sampleName]['thickness']*10**-4
    nu_2_vol = thickness*nu_2
    #return nu
    #return np.mean(nu),np.mean(nu_2),np.mean(nu_3),np.mean(nu_4), area, np.log(maxAvg/minAvg), chargeAvg, chargeAvg_2, chargeAvg_3, chargeAvg_4
    print("CE (cm^2/C): ",  np.mean(nu_2), "+/-", np.std(nu_2))
    print("CE (cm^3/C): ",  np.mean(nu_2_vol), "+/-", np.std(nu_2_vol))
    print("Int_R:" ,int_r, "+/-",int_r_err)
    return np.mean(nu_2), area, int_r, chargeAvg_2, charge_err, thickness, np.mean(nu_2_vol), np.std(nu_2), np.std(nu_2_vol), int_r_err
       
    
def responseTime(transmission, timeInt,sampleRun, plotData = False, saveFig = False):
    """
    Possible methods:
    rolling averages when the gradient doesn't change by much and then continues to not change
    fit an exponential
    """
    cycles = 5
    
    startBlch = [0]*cycles
    startCltn = [0]*cycles
    endCltn = [0]*cycles
    endBlch = [0]*cycles
    
    factor = max(transmission)
    transmissionScaled = [t/factor for t in transmission]
    
    differential = [transmissionScaled[i+1]-transmissionScaled[i] for i in range(len(transmissionScaled)-1)]
    thresholdBleach = 0.003
    thresholdCol = 0.0005
    MattMacFarlane = 3
    
    for n in range(cycles):
        if n == 0:
            startValBleach = 0
            startValCol = 0
        else:
            startValBleach = startBlch[n-1]+10
            startValCol = startCltn[n-1]+10
        
        i=startValBleach
        while differential[i] < 0.05:
            i+=1
        startBlch[n] = i-1
        i=startValCol
        while differential[i] > -0.04:
            i+=1
        startCltn[n] = i-2
        
        i = startBlch[n]
        while abs(np.mean(differential[i:i+MattMacFarlane]))>thresholdBleach:
            i+=1
        endBlch[n] = i
        i = startCltn[n]
        while abs(np.mean(differential[i:i+MattMacFarlane]))>thresholdCol:
            i+=1
        endCltn[n] = i+1
        
    if plotData:
        plt.plot(timeInt[:-1],differential)
        plt.ylabel(r"$\frac{d}{dt}($Transmission$)$ (A.U.)", fontsize=30)
        plt.xlabel("Time (s)", fontsize=30)
        plt.xticks(fontsize = 30)
        plt.yticks(fontsize = 30)
        plt.grid(True)
        fig = plt.gcf()
        fig.set_size_inches(16.5,9.5)
        if saveFig:
            plt.savefig("Transmission/Graphs/"+sampleRun+"_differential.png")
        else:
            plt.show()
        
        plt.plot(timeInt, transmission)
        for n in range(cycles):
            plt.axvline(timeInt[startBlch[n]], color='g')
            plt.axvline(timeInt[endBlch[n]], color='g')
            plt.axvline(timeInt[startCltn[n]], color='r')
            plt.axvline(timeInt[endCltn[n]], color='r')
        plt.ylabel("Transmission (A.U.)", fontsize=30)
        plt.xlabel("Time (s)", fontsize=30)
        plt.xticks(fontsize = 30)
        plt.yticks(fontsize = 30)
        #plt.grid(True)
        fig = plt.gcf()
        fig.set_size_inches(16.5,9.5)
        if saveFig:
            plt.savefig("Transmission/Graphs/"+sampleRun+"_response_times.png")
        else:
            plt.show()
    
    bleachingTimes = [timeInt[endBlch[n]]-timeInt[startBlch[n]] for n in range(1,cycles)]
    colourationTimes = [timeInt[endCltn[n]]-timeInt[startCltn[n]] for n in range(cycles)]
    
    print("Colouration Response Times: ", colourationTimes)
    print("Bleaching Response Times: ", bleachingTimes)
    return colourationTimes, bleachingTimes

if __name__ == "__main__":
    sampleDict = readSampleData()
    sampleDict = readMorphologyData(sampleDict)
    if sampleRun.find("run") != -1: #find the sample name from the run name
        sampleName = sampleRun[:sampleRun.find("run")-1]
    else:
        sampleName = sampleRun
        runsNum = int(sampleDict[sampleName]['transmission_runs'])
        print("Number of runs found for sample:", runsNum)
        sampleRun = sampleName+"_run_1"
    if int(sampleRun[7:7+sampleRun[7:].find("_")]) == 14:
        batchNo="Batch_3"
    elif int(sampleRun[7:7+sampleRun[7:].find("_")]) == 2 or int(sampleRun[7:7+sampleRun[7:].find("_")]) == 5:
        batchNo="Batch_2"
    else:
        batchNo="Batch_1"

    amperometryName = initDir+"Transmission/"+batchNo+"/"+sampleRun+"/"+sampleRun+"_amperometry.csv"  #name of amperometry file
    direc = initDir+"Transmission/"+batchNo+"/"+sampleRun+"/"
    #plotSpectra = True
    transmission, timeInt = readTransmission(direc, sampleRun, sampleDict,sampleName, batchNo, plotSpectra = False)
    current, time = readAmperometry(amperometryName)
    transmission, timeInt = alignData(transmission, timeInt, time)
    nu, area, int_r, chargeAvg, charge_err, thickness, nu_vol, nu_err, nu_vol_err, int_r_err = colourationEfficiency(transmission, timeInt, current, time, sampleDict, sampleName)
    colourationTimes, bleachingTimes = responseTime(transmission, timeInt, sampleRun, plotData = True, saveFig = True)
    if displayPlot == True:
        plotData(transmission, timeInt, current, time, sampleRun, saveFig = False)

