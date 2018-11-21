import os
import sys
import warnings
warnings.simplefilter("ignore")
import csv
import math
import tkinter
import tkinter.filedialog
import tkinter.messagebox
import numpy
from scipy.misc import imresize
from keras.models import load_model
import keras.backend
from matplotlib import pyplot as plt

from loadConfiguration import Configuration
from imagedObject import FileImagedObject
from getObjectHierarchyLabels import getObjectHierarchyLabels





#Returns the resultant image (or layer activation) of a certain filter at a certain layer in a model given the input layer,
#output layer, output layer filter number and input data.
def getLayerActivation(modelToUse,inputLayerName,outputLayerName,filterNumber,inputImageData):
    inputTensor=modelToUse.get_layer(name=inputLayerName,index=0).input
    outputTensor=modelToUse.get_layer(name=outputLayerName).output
    
    outputFunction=keras.backend.function([inputTensor],[outputTensor])
    expandedInput=numpy.expand_dims(inputImageData,axis=0) #Adds a batch axis of size 1 to int input data as the model requires data in this format.
    fullOutputArray=numpy.array(outputFunction([expandedInput,0])) #The zero argument indicates to use the validation mode of the model (no dropout).

    #The first axis is used when multiple pairs of input and output layers are 
    #used in keras.backend.function(), and in this case has a size of 1. The second axis is the batch number, which also has a size
    # of 1 in this case. By using a particular filter number for the final index in the numpy array the result is a 2d image.
    return fullOutputArray[0,0,:,:,filterNumber]   


#Resizes the outputs of getLayerActivation() to a particular size and adds the together
def addLayerActivations(modelToUse,inputLayerName,outputLayerNames,filterNumbers,inputImageData,outputShape):
    if(len(outputLayerNames)!=len(filterNumbers)):
        raise Exception("Each output layer needs an associated filter")
    
    currentSum=numpy.zeros(shape=outputShape)
    
    for i in range(0,len(outputLayerNames)):
        currentActivation=getLayerActivation(modelToUse,inputLayerName,outputLayerNames[i],filterNumbers[i],inputImageData)
        currentActivation=imresize(currentActivation,outputShape)
        currentSum=numpy.add(currentSum,currentActivation)
        
    return currentSum



def createObjectHierarchyLabelString(labels):
    outputString="Labels in object hierarchy:"+"\n"
    
    for currentLevelIndex,currentLevelLabels in enumerate(labels):
        outputString+=("\n"+"Level: "+str(currentLevelIndex))
        
        for currentLabel in currentLevelLabels:
            outputString+=("\n"+"  "+currentLabel)
            
    return outputString


def main():
    gui=tkinter.Tk()
    gui.withdraw() #Hides the main window of the GUI as it is not needed.
    
    input("You will now select the B-CNN model file, press enter to continue")
    modelPath=tkinter.filedialog.askopenfilename()
    print("B-CNN model file "+modelPath+" selected")
    loadedModel=load_model(modelPath)
    
    print("\n")
    input("You will now select an appropiate inputConfiguration.txt file that is compatible with the model, press enter to continue")       
    inputConfigurationFilePath=tkinter.filedialog.askopenfilename()
    print("inputConfiguration file "+inputConfigurationFilePath+" selected")


    inputConfiguration=Configuration(inputConfigurationFilePath,"=")

    allowedFileSuffixes=inputConfiguration.getConfigurationValue("useFileSuffix","raw")
    channelsPerImagedObject=1 if(type(allowedFileSuffixes)==str) else len(allowedFileSuffixes)
    desiredImageSize=inputConfiguration.getConfigurationValue("desiredImageSize","int")
    desiredImageFov=inputConfiguration.getConfigurationValue("desiredImageFov","float")
    contigiousEqualAreaRejectionThreshold=inputConfiguration.getConfigurationValue("contigiousEqualAreaRejectionThreshold","int")
    objectTypeLabels=inputConfiguration.getConfigurationValue("allowedObjectType","raw")
    
    #A map between object types and their corresponding label lists is created.
    objectTypeLabelDictionary={i[0]:tuple(i[1:]) for i in objectTypeLabels}   
    objectTypePossibleLabelSets,objectHierarchyDepth=getObjectHierarchyLabels(list(objectTypeLabelDictionary.values()))


    #The input configuration is printed below
    print("\n")
    print("Input configuration loaded:")
    
    print(" Original file suffixes:")
    for currentSuffix in allowedFileSuffixes:
        print("  "+currentSuffix)
     
    print(" Number of channels for input objects: "+str(channelsPerImagedObject))    
    print(" Image size: "+str(desiredImageSize)+" pixels")
    print(" Image field of view: "+str(desiredImageFov))
    print(" Contgious colour area rejection threshold: "+str(contigiousEqualAreaRejectionThreshold))
    
    print(" Labels at each level in the object type heirarchy:")
    for i in range(0,objectHierarchyDepth):
        print("  Level "+str(i)+":")
    
        for j in objectTypePossibleLabelSets[i]:
            print("   "+j)
    
    
    
    #The input images are now selected
    imageFilePaths=["" for i in range(0,channelsPerImagedObject)]
    print("\n")
    print("\n")    
    print("File selection for an input object will now occur") 
    for i in range(0,channelsPerImagedObject):
        print("\n")
        input("Image "+str(i+1)+" will now be loaded; original training file suffix was "+str(allowedFileSuffixes[i])+". Press enter to continue")
        currentChosenFilePath=tkinter.filedialog.askopenfilename()
        print("File path "+currentChosenFilePath+" was chosen" if(currentChosenFilePath!="") else "No file chosen")
        imageFilePaths[i]=currentChosenFilePath
        
    print("\n")   
    print("The following files were chosen:")
    for currentIndex,currentFilePath in enumerate(imageFilePaths):
        print(" "+allowedFileSuffixes[currentIndex]+" slot: "+currentFilePath)
    input("An ImagedObject will now be created from these files, press enter to continue")
    
    
    
    loadedImagedObject=FileImagedObject(imageFilePaths," ",None,desiredImageSize,desiredImageFov,contigiousEqualAreaRejectionThreshold)
    
    if(loadedImagedObject.nonBlankImageCount==0): #If no images in loadedImagedObject are usable the program will exit when the user is ready.
        print("None of the loaded images are usable due to the following reasons that may not be limited to only one: ")
        print(" Images being rejected due to defects such as a non square shape, being detected as being from the edge of a survey")
        print(" Too many channels did not have an image chosen")
        input("Press enter to exit")
        sys.exit()
    
    



    predictedImage=numpy.expand_dims(loadedImagedObject.imageData,axis=0)
    predictedProbabilities=loadedModel.predict(predictedImage)
    predictedClasses=[currentProbabilites.argmax() for currentProbabilites in predictedProbabilities] #Represents the most likely labels using integers.
    predictedClassesStrings=[objectTypePossibleLabelSets[i][predictedClasses[i]] for i in range(0,objectHierarchyDepth)]     
    
    #Displays the predicted probabilities in ther nerminal and writes them to a file
    outputPredictedProbabilitiesFile=open("predictedProbabilites.txt","w")
    print("\n")
    print("Saving predicted probabilities at location "+os.getcwd()+"/predictedProbabilities.txt")
    print("Predicted label is: "+str(predictedClassesStrings)+", predicted label probabilites are: ")
    outputPredictedProbabilitiesFile.write("Predicted label is:"+str(predictedClassesStrings)+", predicted label probabilites are: ")
    for i in range(0,objectHierarchyDepth):
        print("Label level "+str(i)+":")
        outputPredictedProbabilitiesFile.write("\n"+"Label level "+str(i)+":")
        
        for j,currentLabel in enumerate(objectTypePossibleLabelSets[i]):
            currentProbability=(predictedProbabilities[i][0,j])*100.0
            print("  "+currentLabel+": "+str(currentProbability)+"%")
            outputPredictedProbabilitiesFile.write("\n"+"  "+currentLabel+": "+str(currentProbability)+"%")
            
    outputPredictedProbabilitiesFile.close()       
    
 
     
        
    print("\n")
    print("Plots will now be created and saved")
    #Plots are created below 
    numberOfImages=loadedImagedObject.imageData.shape[2]
    subplotDivision=math.ceil(math.sqrt(numberOfImages)) #Done so the images are arranged in a shape as close to a square as possible.   
        
    #The image channels of loadedImagedObject are shown.    
    imageDataFigure=plt.figure(figsize=(8*subplotDivision,8*subplotDivision)) 
    plt.suptitle("Plot of ImagedObject channels")
    for i in range(0,numberOfImages):
        locationString=str(subplotDivision)+str(subplotDivision)+str(i+1)
        currentimageDataAxes=imageDataFigure.add_subplot(locationString)
        currentimageDataAxes.set_title(allowedFileSuffixes[i]+" channel slot")
        currentimageDataAxes.imshow(loadedImagedObject.imageData[:,:,i],cmap="hot")
    
    print("Saving plot of ImagedObject channels at location "+os.getcwd()+"/imageDataFigure.png")  
    imageDataFigure.savefig("imageDataFigure.png")



    outputLayerNames=["out"+str(i+1)+"LocationHeatmap" for i in range(0,objectHierarchyDepth)] #Each output location heatmap is labeled sequentially from the output location heatmap closest to the input layer.
    totalLocationHeatmap=addLayerActivations(loadedModel,"mainInput",outputLayerNames,predictedClasses,loadedImagedObject.imageData,(loadedImagedObject.imageData.shape[0],loadedImagedObject.imageData.shape[1]))
                        
    
    #The previously created location heatmap is shown.
    heatmapFigure,heatmapAxes=plt.subplots(figsize=(8,8))
    heatmapAxes.set_title("Total location heatmap")
    heatmapAxes.imshow(totalLocationHeatmap)
    print("Saving plot of total location heatmap at location "+os.getcwd()+"/totalLocationHeatmap.png")  
    heatmapFigure.savefig("totalLocationHeatmap.png")
    
    
    
    #The image channels of loadedImagedObject are shown with an overlay of the previously created location heatmap.    
    imageDataLocationHeatmapFigure=plt.figure(figsize=(8*subplotDivision,8*subplotDivision)) 
    
    plt.suptitle("Plot of ImagedObject channels with total location heatmap overlay")
    for i in range(0,numberOfImages):
        locationString=str(subplotDivision)+str(subplotDivision)+str(i+1)
        currentimageDataLocationHeatmapAxes=imageDataLocationHeatmapFigure.add_subplot(locationString)
        currentimageDataLocationHeatmapAxes.set_title(allowedFileSuffixes[i]+" channel slot")
        currentimageDataLocationHeatmapAxes.imshow(loadedImagedObject.imageData[:,:,i],cmap="hot")
        currentimageDataLocationHeatmapAxes.imshow(totalLocationHeatmap,alpha=0.4,cmap="winter")
    
    print("Saving plot of ImaghedObject channels with total location heatmap overlay at location "+os.getcwd()+"/totalLocationHeatmap.png")  
    imageDataLocationHeatmapFigure.savefig("imageDataTotalLocationHeatmapFigure.png")    
    
    
main()


