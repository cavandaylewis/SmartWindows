import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence
from scipy.optimize import curve_fit
import os
from transmissionGraphs import readAmperometry, readCyclicVoltAmperometry
from tqdm import tqdm

#dir = "C://Users/Cavan Day-Lewis/OneDrive - University of Bristol/Important Stuff/Cavan Day-Lewis/Bristol University/Physics Year 4/Year 4 Project/Stress Measurements/sample_29_3_19_2/Calibration/Calibration_670.tif"


#To Do:
# - Pure Sample

def plotData(distance, time, current, timeCurr, v1 = None, v2 = None, voltage = None, ylabel = None):
    if voltage is None:
        cycles = 2
        #voltage = np.array((len(timeCurr)))
        voltage = [v1 for i in range(len(timeCurr))]
        for n in range(cycles):
            start1 = next(x[0] for x in enumerate(timeCurr) if x[1]>30*n)
            end1 = next(x[0] for x in enumerate(timeCurr) if x[1]>30*n+15)
            start2 = end1+1
            if n == cycles-1:
                end2 = len(timeCurr)
            else:
                end2 = next(x[0] for x in enumerate(timeCurr) if x[1]>30*(n+1))
            voltage[start1:end1] = [v1 for i in range(end1-start1)]
            voltage[start2:end2] = [v2 for i in range(end2-start2)]
    
    #plt.subplots(3,1,sharex=True)
    ax1 = plt.subplot(311)
    plt.plot(timeCurr, voltage, 'r-')
    plt.ylabel('Voltage (V)', color='r', fontsize=20)
    plt.yticks(color='r', fontsize=20)
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    ax2 = plt.subplot(312, sharex = ax1)
    #plt.plot(time, current, 'g-')
    plt.plot(timeCurr, np.array(current)/1000, 'g-')
    #plt.ylabel(r'Current ($\mu A$)', color='g', fontsize=20)
    plt.ylabel(r'Current (mA)', color='g', fontsize=20)
    plt.yticks(color='g', fontsize=20)
    ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    if ylabel is None:
        ylabel = "Pixels"
    
    ax3 = plt.subplot(313, sharex = ax1)
    plt.plot(time, distance, 'b-')
    plt.ylabel(ylabel, color='b', fontsize=20)
    plt.xlabel('Time (s)', fontsize = 30)
    #plt.xlim(-1,155)
    plt.xticks(fontsize = 30)
    plt.yticks(color='b', fontsize = 20)
    
    fig = plt.gcf()
    fig.set_size_inches(16.5, 9.5)
    plt.show()

def findCentre(im, displayPlot):
    imarray = np.array(im)
    maxPosInt = np.array(np.unravel_index(imarray.argmax(), imarray.shape))
    imarraySmall = imarray[maxPosInt[0]-15:maxPosInt[0]+15, maxPosInt[1]-15:maxPosInt[1]+15]
    maxVal = np.amax(imarray)
    minVal = np.amin(imarray[maxPosInt[0]-1:maxPosInt[0]+1, maxPosInt[1]-1:maxPosInt[1]+1])
    components = [np.array([1,0]),np.array([-1,0]),np.array([0,1]),np.array([0,-1])]
    adj = np.array([0,0])
    for d in components:
        adj = adj+((imarray[tuple(maxPosInt+d)]-minVal)/(2*(maxVal-minVal)))*d
    maxPos = maxPosInt+adj
    
    if displayPlot == True:
        plt.imshow(imarray, cmap=plt.cm.jet)
        #plt.scatter(maxPos[1], maxPos[0], c='white', marker='x')
        plt.scatter(maxPosInt[1], maxPosInt[0], c='white', marker='x')
        plt.show()
        #plt.imshow(imarraySmall, cmap=plt.cm.jet)
        #plt.scatter(maxPos[1]-maxPosInt[1]+15, maxPos[0]-maxPosInt[0]+15, c='white', marker='x')
        #plt.show()
    return maxPosInt

def calibration(dir):
    #open all calibration files
    files = [q for q in os.listdir(dir+"Calibration/") if q.endswith(".tif")]
    microns = [0.0]*(len(files))
    distance = [0.0]*(len(files))
    x = [0.0]*(len(files))
    y = [0.0]*(len(files))
    i = 0
    for f in files:
        microns[i] = int(f[12:15])*10
        print("Taking Data from: ", microns[i])
        im = Image.open(dir+"Calibration/"+f)
        #find Centres
        x[i], y[i] = findCentre(im, displayPlot = False)
        i+=1
    xInit = x[microns.index(max(microns))]
    yInit = y[microns.index(max(microns))]
    for i in range(len(files)):
        microns[i] = abs(max(microns)-microns[i])
        distance[i] = np.sqrt((x[i]-xInit)**2+(y[i]-yInit)**2)
    microns.reverse()
    distance.reverse()
    
    startFit = next(x[0] for x in enumerate(distance) if x[1]>2)
    endFit = int(len(distance)*0.9)
    
    fit = np.polyfit(microns[startFit:endFit],distance[startFit:endFit],1)
    #print(fit)
    microns = [x+fit[1]/fit[0] for x in microns]
    fit[1] = 0
    fit_fn = np.poly1d(fit)
    residuals = [distance[i]-fit_fn(microns[i]) for i in range(startFit-1,len(microns[startFit-1:]))]
    slope_err = np.std(residuals)/microns[-1]
    plt.plot(microns,distance, 'bx', microns[startFit-1:], fit_fn(microns[startFit-1:]), 'r-')
    microns_err = [5 for m in microns]
    distance_err = [3 for d in distance]
    plt.errorbar(microns,distance, xerr = microns_err,yerr = distance_err, fmt='none', ecolor='b', capsize=2)
    
    #plt.errorbar(depositionTime,int_r_pulsed, yerr = int_r_pulsed_error, fmt='none', ecolor='b')
    print("pixels per micron:", fit[0], "+/-", slope_err)
    #plt.plot(microns, distance)
    plt.xlabel(r"Deflection ($\mu m$)", fontsize = 30)
    plt.ylabel("Pixels", fontsize = 30)
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)
    plt.grid(True)
    fig = plt.gcf()
    fig.set_size_inches(16, 9.5)
    plt.show()
    
    return fit[0], slope_err
    
    
def multistepAmperometryContinuous(dir, voltage, slope, length):
    current, timeCurr = readAmperometry(dir+"Amperometry_"+voltage+"_continuous/amperometry.csv")
    im = Image.open(dir+"Amperometry_"+voltage+"_continuous/amperometry_"+voltage+"_continuous.tif")
    imarray = []
    for i, page in tqdm(enumerate(ImageSequence.Iterator(im))):
        #print(i) #replace with progress bar
        imarray.append(np.array(page))
        #imarray.append(page)
    
    
    #plt.imshow(imarray[1])
    #plt.show()
    
    #identfy when laser appears
    maxIntensity = [0.0]*100
    for i in range(100):
        maxIntensity[i] = np.amax(imarray[i])
        #print(np.amax(imarray[i]))
    start = next(x[0] for x in enumerate(maxIntensity) if x[1]>10)
    
    #track laser position with time using find centre
    numFrames = len(imarray)-start
    FPS = 11.6
    time = [0.0]*numFrames
    distance = [0.0]*numFrames
    x = [0.0]*numFrames
    y = [0.0]*numFrames
    for i in range(len(imarray)-start):
        x[i], y[i] = findCentre(imarray[i+start], displayPlot = False)
        distance[i] = np.sqrt((x[i]-x[0])**2+(y[i]-y[0])**2)
        time[i] = i/float(FPS)
    
    deflection = np.array(distance)*(10**-6)/slope
    radius = (((length/100)**2)/(2*deflection**2))+0.5
    stress = (1/radius)*3.65*10**9
    
    cycles = 5
    stressBleachedArr = [0.0]*cycles
    stressColouredArr = [0.0]*cycles
    for n in range(cycles):
        start1 = next(x[0] for x in enumerate(time) if x[1]>30*n+7.5)
        end1 = next(x[0] for x in enumerate(time) if x[1]>30*n+12.5)
        start2 = next(x[0] for x in enumerate(time) if x[1]>30*n+22.5)
        #if n == cycles-1:
        #    end2 = len(time)
        #else:
        end2 = next(x[0] for x in enumerate(time) if x[1]>30*n+27.5)
        stressBleachedArr[n] = np.mean(stress[start1:end1])
        stressColouredArr[n] = np.mean(stress[start2:end2])
        
    stressBleached = np.mean(stressBleachedArr)
    stressBleached_err = np.std(stressBleachedArr)
    stressColoured = np.mean(stressColouredArr)
    stressColoured_err = np.std(stressColouredArr)
    
    plt.plot(time, stress)
    plt.axhline(y=stressBleached)
    plt.axhline(y=stressColoured)
    print("stress bleached: ", stressBleached, "+/-", stressBleached_err)
    print("stress coloured: ", stressColoured, "+/-", stressColoured_err)
    plt.show()
    
    #plotData(distance, time, current, timeCurr, 0.5, float(voltage))
    #plotData(deflection, time, current, timeCurr, 0.5, float(voltage), ylabel = "Deflection (m)")
    #plotData(radius, time, current, timeCurr, 0.5, float(voltage), ylabel = "Radius (m)")
    #plotData(stress, time, current, timeCurr, 0.5, float(voltage), ylabel = "Stress Change (Pa)")
    
    #plt.plot(time, distance)
    #plt.xlabel("Time (s)")
    #plt.ylabel("Pixels")
    #plt.show()
    
def cyclicVoltametry(dir, slope, length):
    current, voltage = readCyclicVoltAmperometry(dir+"Cyclic Voltametry/Cyclic_Voltametry.csv")
    
    im = Image.open(dir+"Cyclic Voltametry/Cyclic_Voltametry.tif")
    imarray = []
    for i, page in tqdm(enumerate(ImageSequence.Iterator(im))):
        #print(i) #replace with progress bar
        imarray.append(np.array(page))
    
    #identfy when laser appears
    maxIntensity = [0.0]*100
    for i in range(100):
        maxIntensity[i] = np.amax(imarray[i])
        #print(np.amax(imarray[i]))
    start = next(x[0] for x in enumerate(maxIntensity) if x[1]>10)
    
    #track laser position with time using find centre
    numFrames = len(imarray)-start
    FPS = 11.6
    time = [0.0]*numFrames
    distance = [0.0]*numFrames
    x = [0.0]*numFrames
    y = [0.0]*numFrames
    for i in range(len(imarray)-start):
        x[i], y[i] = findCentre(imarray[i+start], displayPlot = False)
        distance[i] = np.sqrt((x[i]-x[0])**2+(y[i]-y[0])**2)
        time[i] = i/float(FPS)
    
    #plot position against time
    cycles = current.shape[1]
    nRows = current.shape[0]
    
    currentR = current.ravel(order = 'F')
    voltageR = voltage.ravel(order = 'F')
    timeCurr = []
    VPS = 0.1
    for k in range(cycles):
        timeTemp = [(k*nRows+i)*VPS for i in range(nRows)]
        timeCurr = timeCurr+timeTemp
    
    deflection = np.array(distance)*(10**-6)/slope
    radius = (((length/100)**2)/(2*deflection**2))+0.5
    stress = (1/radius)*3.65*10**9
    
    #plotData(distance, time, currentR, timeCurr, voltage = voltageR)
    #plotData(stress, time, currentR, timeCurr, voltage = voltageR, ylabel = "Stress Change (Pa)")
    
    
    #print(min(voltageR), max(voltageR), int(len(stress)/6))
    singleVoltage = np.linspace(min(voltageR), max(voltageR), int(len(stress)/6))
    
    plt.plot(singleVoltage,stress[0:int(len(stress)/6)][::-1], "b-")
    plt.ylabel("Stress Change (Pa)", fontsize=30)
    plt.xlabel('Voltage (V)', fontsize = 30)
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)
    plt.grid(True)
    fig = plt.gcf()
    fig.set_size_inches(16.5, 9.5)
    plt.show()    
    
    displayVoltagePlot = False
    if displayVoltagePlot == True:
        nFrames = int(FPS/VPS)*2*cycles
        voltagePixel = [0.0]*nFrames
        i=0
        n=0
        for t in time:
            n = np.floor(t*VPS)
            direction = np.exp(1j*(n+1)*np.pi).real
            if i < nFrames:
                voltagePixel[i] = (VPS*t-(0.5+n))*direction
            i+=1
        voltagePixel = np.array(voltagePixel)
        voltagePixel = voltagePixel.reshape((int(nFrames/(cycles)),cycles))
        distanceReshape = np.array(distance[:nFrames]).reshape(voltagePixel.shape)
        
        fig, ax1 = plt.subplots() #beigin plotting
        for i in range(cycles):
            ax1.plot(voltage[:,0],current[:,i], 'b-')
        ax1.set_xlabel('Voltage (V)')
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel(r'Current ($\mu m$)', color='b', fontsize=30)
        ax1.tick_params('y', colors='b', labelsize=30)

        #plt.axhline(color="black")
        #plt.axvline(color="black")

        ax2 = ax1.twinx() #create axis on the same graph
        for i in range(cycles):
            ax2.plot(voltagePixel[:,i],distanceReshape[:,i], 'g-')
        ax2.set_ylabel('Pixels', color='g', fontsize=30)
        #ax1.set_ylim([ax2.get_ylim()[0]*ax1.get_ylim()[1]/ax2.get_ylim()[1], ax1.get_ylim()[1]]) # change the y axis limit of the transmission plot, so that 0 intensity is the same for both current and transmission
        ax2.tick_params('y', colors='g', labelsize=30)
        
        plt.grid(True)
        fig = plt.gcf()
        fig.set_size_inches(16.5, 9.5)
        #plt.show() #fig.tight_layout() 
        plt.close()
        
        
        plt.plot(voltagePixel[:,1], distanceReshape[:,1])
        plt.grid(True)
        fig = plt.gcf()
        fig.set_size_inches(16.5, 9.5)
        plt.show()
    
def multistepAmperometry(dir, voltage, slope, length):
    current, timeCurr = readAmperometry(dir+"Amperometry_"+voltage+"/amperometry.csv")
    files = [q for q in os.listdir(dir+"Amperometry_"+voltage+"/") if q.endswith(".tif")]
    bleachedArr = [] #[0.0]*(len(files)/2)
    colouredArr = [] #[0.0]*(len(files)/2)
    distance = [0.0]*(len(files))
    x = [0.0]*(len(files))
    y = [0.0]*(len(files))
    i = 0
    for f in files:
        im = Image.open(dir+"Amperometry_"+voltage+"/"+f)
        x[i], y[i] = findCentre(im, displayPlot = False)
        distance[i] = np.sqrt((x[i]-x[0])**2+(y[i]-y[0])**2)
        if f.find("bleached") != -1:
            bleachedArr.append(distance[i])
        else:
            colouredArr.append(distance[i])
        i+=1
    print(bleachedArr, colouredArr)
    diff = np.array([colouredArr])-np.array([bleachedArr])
    
    cycles = 5
    total_charge=[0.0]*cycles
    timeSpacing = timeCurr[-1]-timeCurr[-2]
    for n in range(cycles):
        switchCurr = int((n*30+10)//timeSpacing)+3
        endCurr = int(((n+1)*30)//timeSpacing)+3
        total_charge[n] = np.trapz(current[switchCurr:endCurr], dx = timeSpacing)/1000000
    charge = np.mean(abs(np.array(total_charge)))
    charge_error = np.std(abs(np.array(total_charge)))
    
    deflection = diff*(10**-6)/slope
    radius = (((length/100)**2)/(2*deflection**2))+0.5
    stress = (1/radius)*3.65*10**9
    print("Mean Difference:", np.mean(diff), "Error: ", np.std(diff))
    print("Stress Change (Pa): ", np.mean(stress), "Error: ", np.std(stress))
    print("Charge: ", charge, "Error:", charge_error)

def stressVoltage():
    sample_29_3_19_3_v_5_diff_bleached = [0.0, 1.0, 1.4142135623730951, 1.0, 1.0]
    sample_29_3_19_3_v_5_diff_coloured = [14.7648230602334, 13.416407864998739, 13.0, 10.770329614269007, 10.770329614269007]

    sample_29_3_19_3_v_4_diff_bleached = [0.0, 0.0, 1.0, 0.0, 0.0] 
    sample_29_3_19_3_v_4_diff_coloured = [14.317821063276353, 14.317821063276353, 14.317821063276353, 2.23606797749979, 14.317821063276353]

    sample_29_3_19_3_v_3_diff_bleached = [0.0, 1.0, 1.0, 1.0, 1.0]
    sample_29_3_19_3_v_3_diff_coloured = [14.317821063276353, 14.317821063276353, 14.317821063276353, 14.317821063276353, 14.317821063276353]

    sample_29_3_19_3_v_2_diff_bleached = [0.0, 1.0, 0.0, 1.0, 1.0]
    sample_29_3_19_3_v_2_diff_coloured = [10.198039027185569, 10.44030650891055, 10.44030650891055, 10.44030650891055, 10.44030650891055]

    voltage_arr = [1, 0.9, 0.8, 0.7]
    sample_29_3_19_3_diff_mean = [11.661535318279412,11.701470446121041,13.517821063276353,9.791853012565555]
    sample_29_3_19_3_diff_mean_err = [1.8632918876995892,4.748521977757428,0.39999999999999997,0.4373176987251061]
    sample_29_3_19_3_stress_mean = [3079.6784974415523,3521.537498589341,4038.675971442111,2121.48909761011]
    sample_29_3_19_3_stress_mean_err = [1001.4924039607356,1721.8615171285949,244.1039524576594,191.23958234332085]


    sample_29_3_19_2_v_5_diff_bleached = [0.0, 16.1245154965971, 16.64331697709324, 15.264337522473747, 16.64331697709324] 
    sample_29_3_19_2_v_5_diff_coloured = [16.64331697709324, 3.605551275463989, 2.23606797749979, 3.605551275463989, 4.47213595499958]

    sample_29_3_19_2_v_4_diff_bleached = [0.0, 16.1245154965971, 16.1245154965971, 16.1245154965971, 17.4928556845359] 
    sample_29_3_19_2_v_4_diff_coloured = [15.264337522473747, 16.64331697709324, 2.0, 2.0, 4.123105625617661]

    sample_29_3_19_2_v_3_diff_bleached = [0.0, 1.4142135623730951, 14.212670403551895, 0.0, 0.0] 
    sample_29_3_19_2_v_3_diff_coloured = [12.206555615733702, 12.806248474865697, 12.806248474865697, 12.806248474865697, 10.816653826391969]  

    sample_29_3_19_2_v_2_diff_bleached = [0.0, 9.433981132056603, 1.4142135623730951, 9.433981132056603, 0.0]
    sample_29_3_19_2_v_2_diff_coloured = [1.0, 1.4142135623730951, 8.602325267042627, 1.4142135623730951, 7.211102550927978]

    sample_29_3_19_2_diff_mean = [-6.8225727025473475,-5.16712840982851,9.163014180159553,-0.12806417675390094]
    sample_29_3_19_2_diff_mean_err = [11.769628216087481,11.6406801143494,5.328270596730507,6.829645546519431]
    sample_29_3_19_2_stress_mean = [3939.3737316265797,3452.634362079516,2391.472862934475,993.1994686212605]
    sample_29_3_19_2_stress_mean_err = [1107.6640746413239,1762.7570484675123,1223.4463247438393,500.2774359712146]
    sample_29_3_19_2_charge_mean = [0.013590896972445424, 0.0020100254071857685, 0.0009124778639354491, 0.00046625139875621174]
    sample_29_3_19_2_charge_mean_err = [0.0005545745065611942, 1.9141363090099343e-05, 0.00016064293620170813, 6.32806403885674e-06]
    
    
    ### continuous
    voltage_arr = [-0.2, -0.3, -0.4, -0.5]
    stress_arr = [875.4518072379324-111.93479092346027, 1668.5894896669015-381.72046684048, 1880.351725822517-913.8294395922488,1581.4783602993857-813.9257485003138]
    stress_arr_err = [(2.6443645039111994+2.220021605099534)/2,(10.813283707033815+17.114413748196064)/2,(36.19976316442401+77.41903554886001)/2,(206.10744139950734+174.80103524497144)/2]
    
    
    """
    voltage = -0.2
    stress bleached:  875.4518072379324 +/- 2.6443645039111994
    stress coloured:  111.93479092346027 +/- 2.220021605099534
    
    voltage = -0.3
    stress bleached:  381.72046684048 +/- 10.813283707033815
    stress coloured:  1668.5894896669015 +/- 17.114413748196064
    
    voltage = -0.4
    stress bleached:  913.8294395922488 +/- 36.19976316442401
    stress coloured:  1880.351725822517 +/- 77.41903554886001
    
    voltage = -0.5
    stress bleached:  813.9257485003138 +/- 206.10744139950734
    stress coloured:  1581.4783602993857 +/- 174.80103524497144
    """
    #y = stress_arr[:]
    #yerr = stress_arr_err
    y = sample_29_3_19_2_stress_mean[::-1]
    yerr = sample_29_3_19_2_stress_mean_err[::-1]
    x = voltage_arr[:]
    #x = sample_29_3_19_2_charge_mean
    #xerr = sample_29_3_19_2_charge_mean_err
    fit = np.polyfit(x,y,1)
    fit_fn = np.poly1d(fit)
    residuals = [y[i]-fit_fn(x[i]) for i in range(len(x))]
    slope = fit[0]
    slope_err = np.std(residuals)/(abs(x[-1]-x[0]))
    plt.plot(x,y, "bx")
    plt.plot(x, fit_fn(x), 'r-')
    #plt.errorbar(x,y, yerr = yerr, xerr=xerr,fmt='none', capsize = 2)
    plt.errorbar(x,y, yerr = yerr,fmt='none', capsize = 2)
    print("Slope: ", slope, "+/-", slope_err)
    #print(y[0], yerr[0], y[-1], yerr[-1])

    #plt.xlabel("Charge (C)",fontsize=30)
    plt.xlabel("Voltage (V)",fontsize=30)
    plt.ylabel(r"Stress Change (Pa)",fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.grid(True)
    fig = plt.gcf()
    fig.set_size_inches(16.5,9.5)
    plt.show()
    
def pureSample(dir, slope, length):
    voltage = -0.5
    current, timeCurr = readAmperometry(dir+"amperometry.csv")
    endTime = next(x[0] for x in enumerate(timeCurr) if x[1]>30*2)-1
    current = current[:endTime]
    timeCurr = timeCurr[:endTime]
    im = Image.open(dir+"amperometry_-0.5.tif")
    imarray = []
    for i, page in tqdm(enumerate(ImageSequence.Iterator(im))):
        #print(i) #replace with progress bar
        imarray.append(np.array(page))
        #imarray.append(page)
    
    
    #plt.imshow(imarray[1])
    #plt.show()
    
    #identfy when laser appears
    maxIntensity = [0.0]*100
    for i in range(100):
        maxIntensity[i] = np.amax(imarray[i])
        #print(np.amax(imarray[i]))
    start = next(x[0] for x in enumerate(maxIntensity) if x[1]>10)
    
    #track laser position with time using find centre
    numFrames = len(imarray)-start
    FPS = 11.6
    time = [0.0]*numFrames
    distance = [0.0]*numFrames
    x = [0.0]*numFrames
    y = [0.0]*numFrames
    for i in range(len(imarray)-start):
        x[i], y[i] = findCentre(imarray[i+start], displayPlot = False)
        distance[i] = np.sqrt((x[i]-x[0])**2+(y[i]-y[0])**2)
        time[i] = i/float(FPS)
    
    deflection = np.array(distance)*(10**-6)/slope
    radius = (((length/100)**2)/(2*deflection**2))+0.5
    stress = (1/radius)*3.65*10**9
    
    plotData(distance, time, current, timeCurr, 0.5, float(voltage))
    #plotData(deflection, time, current, timeCurr, 0.5, float(voltage), ylabel = "Deflection (m)")
    #plotData(radius, time, current, timeCurr, 0.5, float(voltage), ylabel = "Radius (m)")
    plotData(stress, time, current, timeCurr, 0.5, float(voltage), ylabel = "Stress Change (Pa)")

    
if __name__ == "__main__":
    sampleName = "sample_29_3_19_3"
    #sampleName = "sample_pure_gold"


    if sampleName == "sample_29_3_19_1" or sampleName == "sample_pure_gold":
        slope = 0.2996152068613213 #average taken from samples 2 and 3
        totLength = 6.125 #average taken from samples 2 and 3
        deposLength = 5.75 #average taken from samples 2 and 3
        intLength = 5.425 #average taken from samples 2 and 3

    if sampleName == "sample_29_3_19_2":
        slope = 0.3035902506921489 
        totLength = 6.1
        deposLength = 5.85
        intLength = 5.55

    if sampleName == "sample_29_3_19_3":
        slope = 0.29564016303049373
        totLength = 6.15
        deposLength = 5.65
        intLength = 5.30
    
    dir = "C://Users/Cavan Day-Lewis/OneDrive - University of Bristol/Important Stuff/Cavan Day-Lewis/Bristol University/Physics Year 4/Year 4 Project/Stress Measurements/"+sampleName+"/"
    #slope, slope_err = calibration(dir)
    #multistepAmperometryContinuous(dir, "-0.2", slope, totLength)
    #multistepAmperometry(dir, "-0.5", slope, totLength)
    #cyclicVoltametry(dir, slope, totLength)
    stressVoltage()
    if sampleName == "sample_pure_gold":
        pureSample(dir, slope, totLength)