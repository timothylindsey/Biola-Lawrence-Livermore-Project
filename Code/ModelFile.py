import pickle
import tensorflow as tf
from sklearn import svm
from keras.models import load_model

#this file loads svms, nn in an easy format

#Loading Models
def load(fileName, verbose=True): #i.e. svmModel.pkl
    form = fileName.split('.')[1] #i.e. pkl or h5
    if(form == "pkl" or form == "PKL"): #svm/svr model
        with open('../Models/' + fileName,'rb') as f:
            model = pickle.load(f)
    elif(form == "h5" or form == "H5"): #tensorflow NN
        model = load_model('../Models/' + fileName)
    if(verbose): #print out model description
        printDescription(fileName)
    return model

def printDescription(fileName): #file is like svmModel.pkl
    textFileName = fileName.split('.')[0] + '.txt' #text file
    with open('../Models/' + textFileName, 'r') as f:
        print("Model Description: ")
        print(f.read())

#Saving Models
def save(model, fileName, description): #i.e. svmModel, svmModel.pkl, "This model does blah blah"
    form = fileName.split('.')[1]
    if(form == "pkl" or form == "PKL"): #svm/svr model
        with open('../Models/' + fileName,'wb') as f:
            pickle.dump(model, f)
    elif(form == "h5" or form == "H5"): #tensorflow NN
        model.save('../Models/' + fileName)
    makeDescription(fileName, description)

def makeDescription(fileName, description): #file is like svmModel.pkl
    textFileName = fileName.split('.')[0] + '.txt' #text file
    with open('../Models/' + textFileName, 'w') as f:
        f.write(description)
        
