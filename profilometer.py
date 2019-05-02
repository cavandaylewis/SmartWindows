import numpy as np
import matplotlib.pyplot as plt
import os

directory = "C://Users/Cavan Day-Lewis/Google Drive/M Sci Project - Smart Windows 2018/Data/Profileometer/CavanFinn/26-03-19/"
directory2 = "C://Users/Cavan Day-Lewis/Google Drive/M Sci Project - Smart Windows 2018/Data/Profileometer/CavanFinn/"

def openCSV(directory):
    with open(directory) as f:
        lines = f.readlines()
    f.close
    
    i=0
    found = False
    while found == False:
        if lines[i].find("Lateral") != -1:
            found = True
        i+=1
    headers = i+1
    
    vals = np.ndarray((len(lines)-headers,2), dtype="object_")
    for i in range(len(lines)-headers-1):
        vals[i,:] = lines[i+headers].rstrip().split(",")[:-2]
    
    lateral = vals[:,0]
    profile = vals[:,1]
    
    return lateral[:-1].astype(np.float), profile[:-1].astype(np.float)
    
def plotData(lateral, profile, sampleRun, regionsOn = None):
    plt.plot(lateral, profile)
    if regionsOn is not None:
        WO3Max = next(x[0] for x in enumerate(lateral) if x[1]>0.6)
        WO3Min = 0 #next(x[0] for x in enumerate(lateral) if x[1]>0)
        FTOMax = len(lateral)-1 #next(x[0] for x in enumerate(lateral) if x[1]>1.5)
        FTOMin = next(x[0] for x in enumerate(lateral) if x[1]>1.1)
        plt.axvspan(lateral[WO3Min],lateral[WO3Max], facecolor = "green", alpha = 0.5)
        plt.axvspan(lateral[FTOMin],lateral[FTOMax], facecolor = "green", alpha = 0.5)
        plt.axhline(np.mean(profile[WO3Min:WO3Max]), color= "r")
        plt.axhline(np.mean(profile[FTOMin:FTOMax]), color= "r")
    
    plt.ylabel(r"Z ($\mu m$)", fontsize=30)
    plt.xlabel("X (mm)", fontsize = 30)
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)
    
    plt.grid(True)
    fig = plt.gcf()
    fig.set_size_inches(16, 9.5)
    #plt.savefig("Transmission/Graphs/"+sampleRun+"_profilometery.png")
    #plt.close()
    plt.show()
    
files = [q for q in os.listdir(directory2) if q.endswith(".csv")]

f = "2019-03-15_S11_002.csv"
ending = f[f.find("S")+1:-4]
sampleRun = "sample_14_3_19_"+ending
lateral, profile = openCSV(directory2+"/"+f)
plotData(lateral, profile, sampleRun, regionsOn = True)

"""
for f in files:
    ending = f[f.find("S")+1:-4]
    sampleRun = "sample_14_3_19_"+ending
    print(sampleRun)
    try:
        lateral, profile = openCSV(directory2+"/"+f)
        plotData(lateral, profile, sampleRun)
    except:
        print("Unable to process: ", sampleRun)


for i in range(18,25):
    for j in range(1,4):
        sampleRun = "sample_14_3_19_"+str(i)+"_00"+str(j)
        print(sampleRun)
        try:
            lateral, profile = openCSV(directory+sampleRun+".csv")
            plotData(lateral, profile, sampleRun)
        except:
            print("Unable to process: ", sampleRun)
"""
