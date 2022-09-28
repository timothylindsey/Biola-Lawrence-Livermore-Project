import csv
import numpy as np


def getTrain(defaultValue = -10e10):
    return combineDataSets("train", defaultValue=defaultValue)

def getTest(defaultValue = -10e10):
    return combineDataSets("test", defaultValue=defaultValue)

def getValidate(defaultValue = -10e10):
    return combineDataSets("validate", defaultValue=defaultValue)


def combineDataSets(dataset, defaultValue = -10e10, defInf = 10e10): #dataset should be "train", "test" or "validate"
    #get descriptor and docking data respectively
    #if you're not on windows, perish "/" reigns superior
    compounds1, labels1, data1 = loadCompoundData("../Data/" + dataset + "DataDescriptors.csv", defValue = defaultValue)
    compounds2, smiles, labels2, data2, activities = loadDockingData("../Data/" + dataset + "DockingData.csv", defValue = defaultValue)
    
    #format and connect data
    
    if(len(compounds1) != len(compounds2)): #check if files have same number and matching compounds
        print("File compounds not proper!")
        return -1
    for i in range(len(compounds1)): #check compound id's match
        if(compounds1[i] != compounds2[i]):
            print("File compound don't match!")
            return -1
        
    compounds = compounds1 #compuonds match
    labels = labels2 + labels1 #docking data then descriptors
    compoundData = np.concatenate((data2,data1), axis=1) #connect the two data arrays together
    compoundData = compoundData.astype(float)
    
    #replace inf with a real number
    compoundData[np.isinf(compoundData)] = defInf
    compoundData[np.isneginf(compoundData)] = -defInf
    
    return compounds, smiles, labels, compoundData, activities



def loadCompoundData(fileName, defValue = -10e10): #load a compound file descriptors, first row should be labels
    csvfile=open(fileName, newline='',  encoding='UTF-8')
    rd = csv.reader(csvfile, delimiter=',')
    
    data=[]
    compounds=[]
    for lv in rd:
        compounds.append(lv[0]) #the compound id
        data.append(lv[1:]) #all the statistics (except id), should all be number format
    
    compounds = np.array(compounds)[1:] #skip first label
    labels = data[0] #first row is just labels
    
    data = data[1:] #skip labels
    data = np.array(data) #np array
    data[data == ''] = defValue #replace empty strings with value
    data = data.astype(np.float) #string -> floats
    
    return compounds, labels, data

def loadDockingData(fileName, defValue = -10e10): #load a docking data file, first row should be labels
    csvfile=open(fileName, newline='',  encoding='UTF-8')
    rd = csv.reader(csvfile, delimiter=',')
    
    docking=[]
    compounds=[] #compound id
    smiles=[] #smile identifier
    activities=[] #activity, what we actually try to match
    for lv in rd:
        smiles.append(lv[0])
        compounds.append(lv[11])
        activities.append(lv[12])
        docking.append(lv[1:11] + lv[13:])
    
    activities = (np.array(activities)[1:]).astype(float) #skip first label and convert from string to floats
    smiles = np.array(smiles)[1:] #skip first label
    compounds = np.array(compounds)[1:] #skip first label
    
    labels = docking[0]
    docking = docking[1:] #skip data row
    
    docking = np.array(docking)
    docking[docking == ''] = defValue #replace empty strings with value
    docking = docking.astype(np.float) #string -> floats
    
    return compounds, smiles, labels, docking, activities