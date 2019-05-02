from transmissionGraphs import readSampleData,readTransmission,readAmperometry,alignData,colourationEfficiency,responseTime,plotData, readMorphologyData, responseTime
import matplotlib.pyplot as plt
import numpy as np

initDir = "C://Users/Cavan Day-Lewis/OneDrive - University of Bristol/Important Stuff/Cavan Day-Lewis/Bristol University/Physics Year 4/Year 4 Project/"

#runs = ['sample_14_3_19_1_run_1', 'sample_14_3_19_1_run_2', 'sample_14_3_19_1_run_3', 'sample_14_3_19_9_run_1', 'sample_14_3_19_9_run_2', 'sample_14_3_19_9_run_3']

#runs = ['sample_14_3_19_1_run_1', 'sample_14_3_19_1_run_2', 'sample_14_3_19_1_run_3', 'sample_14_3_19_2_run_1', 'sample_14_3_19_2_run_2', 'sample_14_3_19_2_run_3', 
#        'sample_14_3_19_3_run_1', 'sample_14_3_19_3_run_2', 'sample_14_3_19_3_run_3', 'sample_14_3_19_4_run_1', 'sample_14_3_19_4_run_2', 'sample_14_3_19_4_run_3', 
#        'sample_14_3_19_5_run_1', 'sample_14_3_19_5_run_2', 'sample_14_3_19_5_run_3', 'sample_14_3_19_6_run_1', 'sample_14_3_19_6_run_2', 'sample_14_3_19_6_run_3',
#        'sample_14_3_19_7_run_1', 'sample_14_3_19_7_run_2', 'sample_14_3_19_7_run_3', 'sample_14_3_19_8_run_1', 'sample_14_3_19_8_run_2', 'sample_14_3_19_8_run_3',
#        'sample_14_3_19_9_run_1', 'sample_14_3_19_9_run_2', 'sample_14_3_19_9_run_3', 'sample_14_3_19_10_run_1', 'sample_14_3_19_10_run_2', 'sample_14_3_19_10_run_3',
#        'sample_14_3_19_11_run_1', 'sample_14_3_19_11_run_2', 'sample_14_3_19_11_run_3', 'sample_14_3_19_12_run_1', 'sample_14_3_19_12_run_2', 'sample_14_3_19_12_run_3',
#        'sample_14_3_19_13_run_1', 'sample_14_3_19_13_run_2', 'sample_14_3_19_13_run_3', 'sample_14_3_19_14_run_1', 'sample_14_3_19_14_run_2', 'sample_14_3_19_14_run_3',
#        'sample_14_3_19_15_run_1', 'sample_14_3_19_15_run_2', 'sample_14_3_19_15_run_3', 'sample_14_3_19_16_run_1', 'sample_14_3_19_16_run_2', 'sample_14_3_19_16_run_3',
runs = ['sample_14_3_19_17_run_1', 'sample_14_3_19_17_run_2', 'sample_14_3_19_17_run_3', 'sample_14_3_19_18_run_1', 'sample_14_3_19_18_run_2', 'sample_14_3_19_18_run_3',
        'sample_14_3_19_19_run_1', 'sample_14_3_19_19_run_2', 'sample_14_3_19_19_run_3', 'sample_14_3_19_20_run_1', 'sample_14_3_19_20_run_2', 'sample_14_3_19_20_run_3',
        'sample_14_3_19_21_run_1', 'sample_14_3_19_21_run_2', 'sample_14_3_19_21_run_3', 'sample_14_3_19_22_run_1', 'sample_14_3_19_22_run_2', 'sample_14_3_19_22_run_3',
        'sample_14_3_19_23_run_1', 'sample_14_3_19_23_run_2', 'sample_14_3_19_23_run_3', 'sample_14_3_19_24_run_1', 'sample_14_3_19_24_run_2', 'sample_14_3_19_24_run_3']




def runAll(runs, initDir):
    with open("transmissionResults.csv", "w") as file:
        file.write("Sample, CE, CE_err, CE_vol, CE_vol_err, int_r, int_r_err, charge, charge_err, area, thickness, bleaching_time, bleaching_time_err, colouration_time, colouration_time_err\n")
    file.close()
    nu = [0.0]*(len(runs))
    nu_err = [0.0]*(len(runs))
    nu_vol = [0.0]*(len(runs))
    nu_vol_err = [0.0]*(len(runs))
    int_r = [0.0]*(len(runs))
    int_r_err = [0.0]*(len(runs))
    chargeAvg = [0.0]*(len(runs))
    charge_err = [0.0]*(len(runs))
    area = [0.0]*(len(runs))
    thickness = [0.0]*(len(runs))
    bleaching_time = [0.0]*(len(runs))
    bleaching_time_err = [0.0]*(len(runs))
    colouration_time = [0.0]*(len(runs))
    colouration_time_err = [0.0]*(len(runs))
    i=0
    for f in runs:
        try:
            sampleRun = f
            print("running: "+sampleRun)
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
            transmission, timeInt = readTransmission(direc, sampleRun, sampleDict, sampleName,batchNo)
            current, time = readAmperometry(amperometryName)
            transmission, timeInt = alignData(transmission, timeInt, time)
            nu[i], area[i], int_r[i], chargeAvg[i], charge_err[i], thickness[i], nu_vol[i], nu_err[i], nu_vol_err[i], int_r_err[i] = colourationEfficiency(transmission, timeInt, current, time, sampleDict, sampleName)
            responseTimes = responseTime(transmission, timeInt, sampleRun, True, True)
            colouration_time[i] = np.mean(responseTimes[0])
            colouration_time_err[i] = np.std(responseTimes[0])
            bleaching_time[i] = np.mean(responseTimes[1])
            bleaching_time_err[i] = np.std(responseTimes[1])

            plotData(transmission, timeInt, current, time, sampleRun, saveFig = True)
            with open("transmissionResults.csv", "a") as file:
                file.write(sampleRun+","+str(nu[i])+","+str(nu_err[i])+","+str(nu_vol[i])+","+str(nu_vol_err[i])+","+str(int_r[i])+","+str(int_r_err[i])+","+str(chargeAvg[i])+","+str(charge_err[i])+","+str(area[i])+","+str(thickness[i])+","+str(bleaching_time[i])+","+str(bleaching_time_err[i])+","+str(colouration_time[i])+","+str(colouration_time_err[i])+"\n")
            file.close()
        except:
            print("Unable to get results for: "+f)
        i+=1
    return runs,nu, nu_err, nu_vol, nu_vol_err, int_r, int_r_err, chargeAvg, charge_err, area, thickness, bleaching_time, bleaching_time_err, colouration_time, colouration_time_err

def readResultsCSV(initDir):
    with open("transmissionResults.csv") as f: 
        lines = f.readlines()
    f.close
    vals = np.ndarray((len(lines),15), dtype="object_")
    for i in range(len(lines)):
        vals[i,:] = lines[i].rstrip().split(",")
    return [vals[:,i].tolist() for i in range(vals.shape[1])]
        
def resultsToDict(results, sampleDict):
    vals=np.array([np.array(xi) for xi in results])
    vals = vals.transpose()
    
    resultsDict = {}
    resultsDict['NonPulsed1'] = {}
    resultsDict['NonPulsed2'] = {}
    resultsDict['NonPulsedBoth'] = {}
    resultsDict['Pulsed'] = {}
    headers = [h[1:] for h in vals[0,1:]]
    headers.append('thickness_err')
    headers.append('deposition_time')
    #print(headers)
    for key in headers:
        resultsDict['NonPulsed1'][key] = []
        resultsDict['NonPulsed2'][key] = []
        resultsDict['NonPulsedBoth'][key] = []
        resultsDict['Pulsed'][key] = []
    
    #create a sampleList
    sampleList = [v[:-6] for v in vals[1:,0].tolist()]
    sampleList = list(dict.fromkeys(sampleList))
    #print(sampleList)
    
    i=0
    reducedVals = np.zeros((len(sampleList),vals.shape[1]+1))
    for s in sampleList:
        j=0
        for v in vals[1:,0].tolist():
            if v.find(s+"_") != -1: #take averages
                reducedVals[i,:-2] = reducedVals[i,:-2]+vals[1+j,1:].astype(float)
            j+=1
        reducedVals[i,:-2] = reducedVals[i,:-2]/3
        reducedVals[i,-2] = sampleDict[s]['thickness_err']
        reducedVals[i,-1] = sampleDict[s]['deposition_time']
        reducedVals[i,9] = sampleDict[s]['thickness'] #take thickness values from file so that they are in micrometers
        #print(reducedVals)
        if int(s[15:]) <= 8:
            #in Non Dep1
            k = 0
            for key in headers:
                resultsDict['NonPulsed1'][key].append(reducedVals[i,k])
                resultsDict['NonPulsedBoth'][key].append(reducedVals[i,k])
                k+=1
            
        elif int(s[15:]) <= 16 and int(s[15:]) > 8:
            #in Non Dep 2
            k = 0
            for key in headers:
                resultsDict['NonPulsed2'][key].append(reducedVals[i,k])
                resultsDict['NonPulsedBoth'][key].append(reducedVals[i,k])
                k+=1
        else:
            #in Dep
            k = 0
            for key in headers:
                resultsDict['Pulsed'][key].append(reducedVals[i,k])
                k+=1
        i+=1
    return resultsDict

#'CE', 'CE_err', 'CE_vol', 'CE_vol_err', 'int_r', 'int_r_err', 'charge', 'charge_err', 'area', 'thickness', 
#'bleaching_time', 'bleaching_time_err', 'colouration_time', 'colouration_time_err'
    
    
def plotCE(resultsDict):
    #plot CE (area) against thickness
    #plot CE (volume) against thickness
    selection = 'Pulsed'
    y = resultsDict[selection]['CE'] #nu
    y_err = resultsDict[selection]['CE_err'] #nu_err
    x = resultsDict[selection]['thickness'] #thickness
    x_err = resultsDict[selection]['thickness_err']
    if any(np.isnan(x)):
        x = x[1:]
        y = y[1:]
        y_err = y_err[1:]
        x_err = x_err[1:]
    fit = np.polyfit(x,y,1)
    fit_fn = np.poly1d(fit)
    plt.plot(x,y, 'bx', label=selection)
    #plt.plot(x, fit_fn(x), 'b-')
    plt.errorbar(x,y,xerr=x_err, yerr = y_err, fmt='none', ecolor='b', capsize=2)
    print("BestCE-Pulsed: ", max(y) , "+/-",y_err[np.argmax(y)], "Thickness:", x[np.argmax(y)], "+/-",x_err[np.argmax(y)])
    
    selection = 'NonPulsed1'
    y = resultsDict[selection]['CE'] #nu
    y_err = resultsDict[selection]['CE_err'] #nu_err
    x = resultsDict[selection]['thickness'] #thickness
    x_err = resultsDict[selection]['thickness_err']
    if any(np.isnan(x)):
        x = x[1:]
        y = y[1:]
        y_err = y_err[1:]
        x_err = x_err[1:]
    fit = np.polyfit(x,y,1)
    fit_fn = np.poly1d(fit)
    plt.plot(x,y, 'rx', label="Constant 1")
    #plt.plot(x, fit_fn(x), 'r-')
    plt.errorbar(x,y,xerr=x_err, yerr = y_err, fmt='none', ecolor='r', capsize=2)
    print("BestCE-Constant: ", max(y) , "+/-",y_err[np.argmax(y)], "Thickness:", x[np.argmax(y)], "+/-",x_err[np.argmax(y)])
    
    
    selection = 'NonPulsed2'
    y = resultsDict[selection]['CE'] #nu
    y_err = resultsDict[selection]['CE_err'] #nu_err
    x = resultsDict[selection]['thickness'] #thickness
    x_err = resultsDict[selection]['thickness_err']
    if any(np.isnan(x)):
        x = x[1:]
        y = y[1:]
        y_err = y_err[1:]
        x_err = x_err[1:]
    fit = np.polyfit(x,y,1)
    fit_fn = np.poly1d(fit)
    plt.plot(x,y, 'gx', label="Constant 2")
    #plt.plot(x, fit_fn(x), 'g-')
    plt.errorbar(x,y,xerr=x_err, yerr = y_err, fmt='none', ecolor='g', capsize=2)
    
    plt.xlabel(r'Thickness $(\mu m)$', fontsize=30)
    plt.ylabel(r'$\eta$ ($\frac{cm^2}{C}$)', fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=20)
    plt.grid(True)
    fig = plt.gcf()
    fig.set_size_inches(16.5,9.5)
    plt.show()
    
    selection = 'Pulsed'
    y = resultsDict[selection]['CE_vol'] #nu
    y_err = resultsDict[selection]['CE_vol_err'] #nu_err
    x = resultsDict[selection]['thickness'] #thickness
    x_err = resultsDict[selection]['thickness_err']
    if any(np.isnan(x)):
        x = x[1:]
        y = y[1:]
        y_err = y_err[1:]
        x_err = x_err[1:]
    fit = np.polyfit(x,y,1)
    fit_fn = np.poly1d(fit)
    plt.plot(x,y, 'bx', label=selection)
    plt.plot(x, fit_fn(x), 'b-')
    plt.errorbar(x,y,xerr=x_err, yerr = y_err, fmt='none', ecolor='b', capsize=2)
    plt.xlabel(r'Thickness $(\mu m)$', fontsize=30)
    plt.ylabel(r"$\eta '$ ($\frac{cm^3}{C}$)", fontsize=30)
    
    selection = 'NonPulsed1'
    y = resultsDict[selection]['CE_vol'] #nu
    y_err = resultsDict[selection]['CE_vol_err'] #nu_err
    x = resultsDict[selection]['thickness'] #thickness
    x_err = resultsDict[selection]['thickness_err']
    if any(np.isnan(x)):
        x = x[1:]
        y = y[1:]
        y_err = y_err[1:]
        x_err = x_err[1:]
    fit = np.polyfit(x[:-1],y[:-1],1)
    fit_fn = np.poly1d(fit)
    plt.plot(x,y, 'rx', label="Constant 1")
    plt.plot(x[:-1], fit_fn(x[:-1]), 'r-')
    plt.errorbar(x,y,xerr=x_err, yerr = y_err, fmt='none', ecolor='r', capsize=2)
    
    selection = 'NonPulsed2'
    y = resultsDict[selection]['CE_vol'] #nu
    y_err = resultsDict[selection]['CE_vol_err'] #nu_err
    x = resultsDict[selection]['thickness'] #thickness
    x_err = resultsDict[selection]['thickness_err']
    if any(np.isnan(x)):
        x = x[1:]
        y = y[1:]
        y_err = y_err[1:]
        x_err = x_err[1:]
    fit = np.polyfit(x[:-1],y[:-1],1)
    fit_fn = np.poly1d(fit)
    plt.plot(x,y, 'gx', label="Constant 2")
    plt.plot(x[:-1], fit_fn(x[:-1]), 'g-')
    plt.errorbar(x,y,xerr=x_err, yerr = y_err, fmt='none', ecolor='g', capsize=2)
    
    plt.xlabel(r'Thickness $(\mu m)$', fontsize=30)
    plt.ylabel(r"$\eta '$ ($\frac{cm^3}{C}$)", fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=20)
    plt.grid(True)
    fig = plt.gcf()
    fig.set_size_inches(16.5,9.5)
    plt.show()
    
def plotResponseTimes(resultsDict):
    #plot response times against thickness
    
    selection = 'NonPulsedBoth'
    y = resultsDict[selection]['bleaching_time'] #bleaching_time
    y_err = resultsDict[selection]['bleaching_time_err'] #bleaching_time_err
    x = resultsDict[selection]['thickness'] #thickness
    x_err = resultsDict[selection]['thickness_err']
    if any(np.isnan(x)):
        x = x[1:]
        y = y[1:]
        y_err = y_err[1:]
        x_err = x_err[1:]
    fit = np.polyfit(x,y,1)
    fit_fn = np.poly1d(fit)
    plt.plot(x,y, 'bx', label = "Bleaching")
    plt.plot(x, fit_fn(x), 'b-')
    plt.errorbar(x,y,xerr=x_err, yerr = y_err, fmt='none', ecolor='b', capsize=2)
    print("Bleaching-Constant: ", min(y) , "+/-",y_err[np.argmin(y)], "Thickness:", x[np.argmin(y)], "+/-",x_err[np.argmin(y)])
    
    y = resultsDict[selection]['colouration_time'] #colouration_time
    y_err = resultsDict[selection]['colouration_time_err'] #colouration_time_err
    x = resultsDict[selection]['thickness'] #thickness
    x_err = resultsDict[selection]['thickness_err']
    if any(np.isnan(x)):
        x = x[1:]
        y = y[1:]
        y_err = y_err[1:]
        x_err = x_err[1:]
    fit = np.polyfit(x,y,1)
    fit_fn = np.poly1d(fit)
    plt.plot(x,y, 'rx', label = "Colouration")
    plt.plot(x, fit_fn(x), 'r-')
    plt.errorbar(x,y,xerr=x_err, yerr = y_err, fmt='none', ecolor='r', capsize=2)
    print("Colouration-Constant: ", min(y) , "+/-",y_err[np.argmin(y)], "Thickness:", x[np.argmin(y)], "+/-",x_err[np.argmin(y)])
    
    plt.xlabel(r'Thickness $(\mu m)$', fontsize=30)
    plt.ylabel('Response Time (s)', fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=20)
    plt.grid(True)
    fig = plt.gcf()
    fig.set_size_inches(17,9.5)
    plt.show()
    
    selection = 'Pulsed'
    y = resultsDict[selection]['bleaching_time'] #bleaching_time
    y_err = resultsDict[selection]['bleaching_time_err'] #bleaching_time_err
    x = resultsDict[selection]['thickness'] #thickness
    x_err = resultsDict[selection]['thickness_err']
    if any(np.isnan(x)):
        x = x[1:]
        y = y[1:]
        y_err = y_err[1:]
        x_err = x_err[1:]
    fit = np.polyfit(x,y,1)
    fit_fn = np.poly1d(fit)
    plt.plot(x,y, 'bx', label = "Bleaching")
    plt.plot(x, fit_fn(x), 'b-')
    plt.errorbar(x,y,xerr=x_err, yerr = y_err, fmt='none', ecolor='b', capsize=2)
    print("Bleaching-Pulsed: ", min(y), "+/-",y_err[np.argmin(y)], "Thickness:", x[np.argmin(y)], "+/-",x_err[np.argmin(y)])
    
    y = resultsDict[selection]['colouration_time'] #colouration_time
    y_err = resultsDict[selection]['colouration_time_err'] #colouration_time_err
    x = resultsDict[selection]['thickness'] #thickness
    x_err = resultsDict[selection]['thickness_err']
    if any(np.isnan(x)):
        x = x[1:]
        y = y[1:]
        y_err = y_err[1:]
        x_err = x_err[1:]
    fit = np.polyfit(x,y,1)
    fit_fn = np.poly1d(fit)
    plt.plot(x,y, 'rx', label = "Colouration")
    plt.plot(x, fit_fn(x), 'r-')
    plt.errorbar(x,y,xerr=x_err, yerr = y_err, fmt='none', ecolor='r', capsize=2)
    print("Colouration-Pulsed: ", min(y), "+/-",y_err[np.argmin(y)], "Thickness:", x[np.argmin(y)], "+/-",x_err[np.argmin(y)])
    
    plt.xlabel(r'Thickness $(\mu m)$', fontsize=30)
    plt.ylabel('Response Time (s)', fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=20)
    plt.grid(True)
    fig = plt.gcf()
    fig.set_size_inches(17,9.5)
    plt.show()
    
def plotIntR(resultsDict):
    #plot int_r against thickness
    
    selection = 'Pulsed'
    y = resultsDict[selection]['int_r'] #int_r
    y_err = resultsDict[selection]['int_r_err'] #int_r_err
    x = resultsDict[selection]['thickness'] #thickness
    x_err = resultsDict[selection]['thickness_err']
    if any(np.isnan(x)):
        x = x[1:]
        y = y[1:]
        y_err = y_err[1:]
        x_err = x_err[1:]
    fit = np.polyfit(x,y,1)
    fit_fn = np.poly1d(fit)
    plt.plot(x,y, 'bx', label=selection)
    plt.plot(x, fit_fn(x), 'b-')
    plt.errorbar(x,y,xerr = x_err, yerr = y_err, fmt='none', ecolor='b', capsize=2)
    
    selection = 'NonPulsed1'
    y = resultsDict[selection]['int_r'] #int_r
    y_err = resultsDict[selection]['int_r_err'] #int_r_err
    x = resultsDict[selection]['thickness'] #thickness
    x_err = resultsDict[selection]['thickness_err']
    if any(np.isnan(x)):
        x = x[1:]
        y = y[1:]
        y_err = y_err[1:]
        x_err = x_err[1:]
    fit = np.polyfit(x[:-1],y[:-1],1)
    fit_fn = np.poly1d(fit)
    plt.plot(x,y, 'rx', label="Constant 1")
    plt.plot(x[:-1], fit_fn(x[:-1]), 'r-')
    plt.errorbar(x,y,xerr = x_err, yerr = y_err, fmt='none', ecolor='r', capsize=2)
    
    selection = 'NonPulsed2'
    y = resultsDict[selection]['int_r'] #int_r
    y_err = resultsDict[selection]['int_r_err'] #int_r_err
    x = resultsDict[selection]['thickness'] #thickness
    x_err = resultsDict[selection]['thickness_err']
    if any(np.isnan(x)):
        x = x[1:]
        y = y[1:]
        y_err = y_err[1:]
        x_err = x_err[1:]
    #fit = np.polyfit(x[:-1],y[:-1],2)
    fit = np.polyfit(x[:-1],y[:-1],1)
    fit_fn = np.poly1d(fit)
    plt.plot(x,y, 'gx', label="Constant 2")
    xLob = np.linspace(x[0],x[-2],1000)
    plt.plot(xLob, fit_fn(xLob), 'g-')
    plt.errorbar(x,y,xerr = x_err, yerr = y_err, fmt='none', ecolor='g', capsize=2)
    
    plt.xlabel(r'Thickness $(\mu m)$', fontsize=30)
    plt.ylabel(r'$\Delta$OD (A.U.)', fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=20)
    plt.grid(True)
    fig = plt.gcf()
    fig.set_size_inches(16.5,9.5)
    plt.show()
    
def plotCharge(resultsDict):
    #plot charge (per unit volume)
    #plot charge (per unit area)
    
    selection = 'Pulsed'
    a = np.array(resultsDict[selection]['charge'])
    a_err = np.array(resultsDict[selection]['charge_err'])
    b = np.array(resultsDict[selection]['area'])
    y = a/b
    y_err = a_err/b
    x = resultsDict[selection]['thickness'] #thickness
    x_err = resultsDict[selection]['thickness_err']
    if any(np.isnan(x)):
        x = x[1:]
        y = y[1:]
        y_err = y_err[1:]
        x_err = x_err[1:]
    fit = np.polyfit(x,y,1)
    fit_fn = np.poly1d(fit)
    plt.plot(x,y, 'bx', label=selection)
    plt.plot(x, fit_fn(x), 'b-')
    plt.errorbar(x,y,xerr=x_err, yerr = y_err, fmt='none', ecolor='b', capsize=2)
    
    selection = 'NonPulsed1'
    a = np.array(resultsDict[selection]['charge'])
    a_err = np.array(resultsDict[selection]['charge_err'])
    b = np.array(resultsDict[selection]['area'])
    y = a/b
    y_err = a_err/b
    x = resultsDict[selection]['thickness'] #thickness
    x_err = resultsDict[selection]['thickness_err']
    if any(np.isnan(x)):
        x = x[1:]
        y = y[1:]
        y_err = y_err[1:]
        x_err = x_err[1:]
    fit = np.polyfit(x,y,1)
    fit_fn = np.poly1d(fit)
    plt.plot(x,y, 'rx', label="Constant 1")
    plt.plot(x, fit_fn(x), 'r-')
    plt.errorbar(x,y,xerr=x_err, yerr = y_err, fmt='none', ecolor='r', capsize=2)
    
    selection = 'NonPulsed2'
    a = np.array(resultsDict[selection]['charge'])
    a_err = np.array(resultsDict[selection]['charge_err'])
    b = np.array(resultsDict[selection]['area'])
    y = a/b
    y_err = a_err/b
    x = resultsDict[selection]['thickness'] #thickness
    x_err = resultsDict[selection]['thickness_err']
    if any(np.isnan(x)):
        x = x[1:]
        y = y[1:]
        y_err = y_err[1:]
        x_err = x_err[1:]
    fit = np.polyfit(x,y,1)
    fit_fn = np.poly1d(fit)
    plt.plot(x,y, 'gx', label="Constant 2")
    plt.plot(x, fit_fn(x), 'g-')
    plt.errorbar(x,y,xerr=x_err, yerr = y_err, fmt='none', ecolor='g', capsize=2)
    
    plt.xlabel(r"Thickness $(\mu m)$", fontsize=30)
    plt.ylabel(r'Charge per unit area $(\frac{C}{m^2})$', fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=20)
    plt.grid(True)
    fig = plt.gcf()
    fig.set_size_inches(16.5,9.5)
    plt.show()
    
    selection = 'Pulsed'
    a = np.array(resultsDict[selection]['charge'])
    a_err = np.array(resultsDict[selection]['charge_err'])
    b = np.array(resultsDict[selection]['thickness'])
    c = np.array(resultsDict[selection]['area'])
    y = a/(b*c) #chargeAvg
    y_err = a_err/(b*c) #charge_err
    x = resultsDict[selection]['thickness'] #thickness
    x_err = resultsDict[selection]['thickness_err']
    if any(np.isnan(x)):
        x = x[1:]
        y = y[1:]
        y_err = y_err[1:]
        x_err = x_err[1:]
    fit = np.polyfit(x,y,1)
    fit_fn = np.poly1d(fit)
    plt.plot(x,y, 'bx', label=selection)
    plt.plot(x, fit_fn(x), 'b-')
    plt.errorbar(x,y,xerr=x_err, yerr = y_err, fmt='none', ecolor='b', capsize=2)
    
    selection = 'NonPulsedBoth'
    a = np.array(resultsDict[selection]['charge'])
    a_err = np.array(resultsDict[selection]['charge_err'])
    b = np.array(resultsDict[selection]['thickness'])
    c = np.array(resultsDict[selection]['area'])
    y = a/(b*c) #chargeAvg
    y_err = a_err/(b*c) #charge_err
    x = resultsDict[selection]['thickness'] #thickness
    x_err = resultsDict[selection]['thickness_err']
    if any(np.isnan(x)):
        x = x[1:]
        y = y[1:]
        y_err = y_err[1:]
        x_err = x_err[1:]
    fit = np.polyfit(x,y,1)
    fit_fn = np.poly1d(fit)
    plt.plot(x,y, 'rx', label=selection)
    plt.plot(x, fit_fn(x), 'r-')
    plt.errorbar(x,y,xerr=x_err, yerr = y_err, fmt='none', ecolor='r', capsize=2)
    
    plt.xlabel(r"Thickness $(\mu m)$", fontsize=30)
    plt.ylabel(r'Charge per unit volume $(\frac{C}{m^3})$', fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.grid(True)
    plt.legend(fontsize=20)
    fig = plt.gcf()
    fig.set_size_inches(16.5,9.5)
    plt.show()
    
    
#results = runAll(runs, initDir)
results = readResultsCSV(initDir)
sampleDict = readSampleData()
sampleDict = readMorphologyData(sampleDict)
resultsDict = resultsToDict(results, sampleDict)
#print(resultsDict['NonDep1']['CE'], resultsDict['NonDep2']['CE'], resultsDict['NonDepBoth']['CE'], resultsDict['Dep']['CE'])

#plotCE(resultsDict)
plotResponseTimes(resultsDict)
#plotIntR(resultsDict)
#plotCharge(resultsDict)