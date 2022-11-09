#this class performs transformations on the data PCA, normalization, etc.
import numpy as np
from sklearn.decomposition import PCA

### PCA ----------------------------------------------------------------------------------

#for all the following ranges begin is inclusive and end is not inclusive
#[0:10] docking_score_? (0-9)
#[10:20] fusion_score_? (0-9)
#[47:59] chi?? (12, odd labels)
#[65:79] paoe_vsa? (1-14)
#[79:89] smr_vsa? (1-10)
#[89:101] slogp_vsa? (1-12)
#[102:113] estate_vsa? (1-11)
#[113:123] vsa_estate? (1-10)
#[143:228] fr_some_chemical? (85 total)

#if an endDims is not positive (-1 or 0) do not apply the PCA for this range
def applyPCA(labels, trainX, testX, valX, endDims=[3,3,3,3,3,3,3,5], muted = False): #apply PCA to data
    #pca info, format: [endSize, beginLabel, endLabel (inclusive), baseName]
    #endSize better be a smaller dimension than the incoming dimensions
    bcutInfo =    [endDims[0], "bcut2d_mwhi", "bcut2d_mrlow", "bcut2d"]
    chiInfo =     [endDims[1], "chi0", "chi4v", "chi"]
    paoeInfo =    [endDims[2], "peoe_vsa1","peoe_vsa9", "paoe"]
    smrInfo =     [endDims[3], "smr_vsa1","smr_vsa9", "smr"]
    slogpInfo =   [endDims[4], "slogp_vsa1","slogp_vsa9", "slogp"]
    estateInfo =  [endDims[5], "estate_vsa1","estate_vsa9", "estate_vsa"]
    vsaInfo =     [endDims[6], "vsa_estate1","vsa_estate9", "vsa_estate"]
    frInfo =      [endDims[7], "fr_al_coo","fr_urea", "fr"]
    
    #copy all the data and labels
    newTrain = trainX.copy()
    newTest = testX.copy()
    newVal = valX.copy()
    newLabels = np.array(labels.copy())
    labels = list(labels)
    
    allInfo = [bcutInfo,chiInfo, paoeInfo, smrInfo, slogpInfo, estateInfo, vsaInfo, frInfo]
    for info in allInfo: #PCA each section, and fill in the data
        
        if(info[0] < 1): #don't apply PCA
            continue
        
        #find ranges based on labels
        begin = labels.index(info[1])
        end = labels.index(info[2])+1 #should be the label after as end is not inclusive
        
        #apply pca and transform all the data
        pca = PCA(n_components=info[0]) #the number of ending dimensions
        pca.fit(trainX[:,begin:end]) #figure out new dimensions, fit only on the train, and apply to all datasets
        newTrain[:,begin:begin+info[0]] = pca.transform(trainX[:,begin:end]) #replace data, note unneded columns remain
        newTest[:,begin:begin+info[0]] = pca.transform(testX[:,begin:end]) #transform based on training fit
        newVal[:,begin:begin+info[0]] = pca.transform(valX[:,begin:end]) #transform based on training fit
        if(not muted):
            print(info[3], "retention:", pca.explained_variance_ratio_)
            print("\ttotal:", str(sum(pca.explained_variance_ratio_)*100)+'%')
        
        for i in range(0, info[0]): #set new labels
            newLabels[begin+i] = info[3] + '_' + str(i) #new label, numbered
        newLabels[begin+info[0]:end] = "REMOVE" #remaining columns change so we can delete this column later
        
    #now remove all unneeded columns that we didn't use, should be labeled REMOVE
    #i.e. if chi takes columns 55-75 and we reduce it to 4 dimensions, columns 59-75 are junk
    newTrain = newTrain[:,newLabels!="REMOVE"] #use only the columns not labeled REMOVE
    newTest = newTest[:,newLabels!="REMOVE"]
    newVal = newVal[:,newLabels!="REMOVE"]
    newLabels = newLabels[newLabels!="REMOVE"]
    
    return newLabels, newTrain, newTest, newVal

### PCA -------------------------------------------------------------------------- END ---


### fusion and docking -------------------------------------------------------------------

#use only the max fusion and docking score for each entry (note makes the end value be positive)
#fusion and docking data can be anywhere but each must be in a group together in order e.g. docking_score_0 ... docking_score_9
def useMaxFD(labels, X):
    newX = X.copy()
    newLabels = np.array(labels.copy())
    
    #find ranges for fusion and docking scores
    labels = list(labels) #needed for index function
    beginD = labels.index("docking_score_0")
    endD = labels.index("docking_score_9") + 1 #end is not inclusive
    beginF = labels.index("fusion_score_0")
    endF = labels.index("fusion_score_9") + 1 #end is not inclusive
    
    #fill in data and adjust labels
    newX[:,beginD] = np.max(abs(X[:,beginD:endD]),axis=1) #note this will return the magnitude instead of value (always positive)
    newLabels[beginD] = "docking_score_max"
    newLabels[beginD+1:endD] = "REMOVE"
    
    newX[:,beginF] = np.max(abs(X[:,beginF:endF]),axis=1) #note this will return the magnitude instead of value (always positive)
    newLabels[beginF] = "fusion_score_max"
    newLabels[beginF+1:endF] = "REMOVE"
    
    #now remove unneeded columns (e.g. docking_score_1 through 9)
    newX = newX[:,newLabels!="REMOVE"]
    newLabels = newLabels[newLabels!="REMOVE"]
    
    return newLabels, newX

#use the average of the fusion and docking scores
def useAverageFD(labels, X):
    newX = X.copy()
    newLabels = np.array(labels.copy())
    
    #find ranges for fusion and docking scores
    labels = list(labels) #needed for index function
    beginD = labels.index("docking_score_0")
    endD = labels.index("docking_score_9") + 1 #end is not inclusive
    beginF = labels.index("fusion_score_0")
    endF = labels.index("fusion_score_9") + 1 #end is not inclusive
    
    #fill in data and adjust labels
    newX[:,beginD] = np.mean(X[:,beginD:endD],axis=1) #the mean of each row
    newLabels[beginD] = "docking_score_average"
    newLabels[beginD+1:endD] = "REMOVE"
    
    newX[:,beginF] = np.mean(X[:,beginF:endF],axis=1) #the mean of each row
    newLabels[beginF] = "fusion_score_average"
    newLabels[beginF+1:endF] = "REMOVE"
    
    #now remove unneeded columns (e.g. docking_score_1 through 9)
    newX = newX[:,newLabels!="REMOVE"]
    newLabels = newLabels[newLabels!="REMOVE"]
    
    return newLabels, newX

### fusion and docking ----------------------------------------------------------- END ---

### normalizing --------------------------------------------------------------------------

def normalizeData(train,test,val, newMean=0, newStd=1): #normalization is only based on the test data
    oldMean = np.mean(train, axis=0) #array of means, one for each col
    oldStd = np.std(train, axis=0) #array of std's, one for each col
    
    oldStd[oldStd==0] = 1 #remove 0's in the std's, this should essentially never happen
    
    #shift data to fit mean and std (essentially to 0 + new mean, then multiple to get std=1, then multiply to new std
    newTrain = newStd*(((train - oldMean) + newMean) / oldStd)
    newTest = newStd*(((test - oldMean) + newMean) / oldStd)
    newVal = newStd*(((val - oldMean) + newMean) / oldStd)
    
    return newTrain, newTest, newVal

def setMeanZero(train,test,val): #shift the mean to zero, used for PCA
    oldMean = np.mean(train, axis=0) #array of means
    
    newTrain = (train - oldMean)
    newTest = (test - oldMean)
    newVal = (val - oldMean)
    return newTrain, newTest, newVal

### normalizing ------------------------------------------------------------------ END ---

### classification -----------------------------------------------------------------------

def toClassification(Y, point=4): # The resulting array will contain values of -1 if it is below 4 and 1 if it is above
    Y = np.array(Y)
    newY = np.copy(Y)
    newY[Y > point] = 1
    newY[Y <= point] = -1
    
    return newY

def toBinaryClassification(Y, point=4): # The resulting array will contain values of -1 if it is below 4 and 1 if it is above
    Y = np.array(Y)
    newY = np.copy(Y)
    newY[Y > point] = 1
    newY[Y <= point] = 0
    
    return newY

### classification --------------------------------------------------------------- END ---