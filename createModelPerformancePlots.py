import math
import numpy
from matplotlib import pyplot as plt



def createAccuracyPlots(history,outputNames,fileNamePrefix):
    for currentName in outputNames: #Creates a seperate plot for every accuracy output type.
        figure=plt.figure()
        currentValKey="val_"+currentName+"_acc"
        currentTrainKey=currentName+"_acc"
        
        plt.plot(history[currentValKey],label="Validation")
        plt.plot(history[currentTrainKey],label="Training")
        plt.title("Accuracy for output "+currentName+" vs training epoch")
        plt.xlabel("epoch")
        plt.ylabel("Accuracy fraction")
        plt.legend()
        
        figure.savefig(fileNamePrefix+"AccuracyPlot_"+currentName+".png")
         

def createConfusionMatricies(model,testObjects,testObjectImageList,imageSaveFilePath,objectHierarchyLabels):      
    numberOfValidationObjects=len(testObjectImageList)
    #testObjectImageList is turned into a numpy array with the first axis indexing each object.
    validationImageDataArray=numpy.empty(shape=(numberOfValidationObjects,)+testObjectImageList[0].shape)
    for i,currentImageData in enumerate(testObjectImageList):
        validationImageDataArray[i,:,:,:]=currentImageData


    predictedProbabilities=model.predict(x=validationImageDataArray,batch_size=32)


    predictedClassStrings=[] #Holds a list the labels for each object.
    for i in range(0,numberOfValidationObjects): #Loops through each object.
        currentObjectPredictedProbabilities=[currentLevelPredictedProbabilies[i,:] for currentLevelPredictedProbabilies in predictedProbabilities]
        currentObjectPredictedClasses=[currentProbabilites.argmax() for currentProbabilites in currentObjectPredictedProbabilities] #Represents the most likely labels using integers.
        predictedClassStrings.append(currentObjectPredictedClasses)
        
    objectHierarchyDepth=len(predictedProbabilities)  
    subplotDivision=math.ceil(math.sqrt(objectHierarchyDepth)) #Number calculated so that the confusion matrix plots can be arranged in a shape as close to a square as possible.
    figure=plt.figure(figsize=(8*subplotDivision,8*subplotDivision))
    plt.suptitle("Confusion matricies; frequency of predicted labels per true labels")



    #Confusion matricies are filled with correct values
    for i in range(0,objectHierarchyDepth): #Loops through all hierarchy levels.
        currentHierarchyLevelLabelCount=(predictedProbabilities[i]).shape[1]
        currentConfusionMatrix=numpy.zeros(shape=(currentHierarchyLevelLabelCount,currentHierarchyLevelLabelCount)) #Creates a square matrix of the correct size.
        trueLabelCount=[0 for i in range(0,currentHierarchyLevelLabelCount)] #Hold the amount of occurences of each true label.
        
        for j in range(0,numberOfValidationObjects):
            trueLabelString=(testObjects[j].label)[i]
            trueLabelInteger=(objectHierarchyLabels[i]).index(trueLabelString)
            predictedLabelInteger=predictedClassStrings[j][i]
            
            trueLabelCount[trueLabelInteger]+=1
            currentConfusionMatrix[trueLabelInteger,predictedLabelInteger]+=1
        
        
        #Another confusion matrix is created but with fractions of predicted labels to every true label.
        currentConfusionMatrixFraction=numpy.zeros_like(currentConfusionMatrix) 
        for trueLabelIndex in range(0,currentHierarchyLevelLabelCount):
            for predictedLabelIndex in range(0,currentHierarchyLevelLabelCount):
                #Columns of true matrix labels are normalised so that each entry in a column represents
                #the fraction of objects with that true label having that predicted label.
                currentConfusionMatrixFraction[trueLabelIndex][predictedLabelIndex]=float(currentConfusionMatrix[trueLabelIndex][predictedLabelIndex])/float(trueLabelCount[trueLabelIndex])
                 
        
        #Below creates plots for the confusion matricies.  
        currentAxes=figure.add_subplot(subplotDivision,subplotDivision,i+1) 
        plt.matshow(currentConfusionMatrixFraction,cmap="spring",fignum=False,vmin=0.0,vmax=1.0)         
        
        currentAxes.set_xticks(range(0,currentHierarchyLevelLabelCount)) #Done to prevent labels from being assined negative positions on the plot (which is what happens by default).
        currentAxes.set_yticks(range(0,currentHierarchyLevelLabelCount)) #Done to prevent labels from being assined negative positions on the plot (which is what happens by default).
        currentAxes.set_xticklabels(objectHierarchyLabels[i])
        currentAxes.set_yticklabels(objectHierarchyLabels[i])
        currentAxes.xaxis.set_ticks_position("bottom") #Done so the horizontal axis labels don't interfere with the title.
        currentAxes.set_xlabel("Predicted label")
        currentAxes.set_ylabel("True label")
        currentAxes.set_title("Hierarchy level "+str(i+1))                     
        
        for trueLabelIndex in range(0,currentHierarchyLevelLabelCount):
            for predictedLabelIndex in range(0,currentHierarchyLevelLabelCount):
                matrixElementLabelPercentage=str(round(currentConfusionMatrixFraction[trueLabelIndex][predictedLabelIndex]*100.0,1))+"%"
                matrixElementLabelString=matrixElementLabelPercentage+"\n"+str(currentConfusionMatrix[trueLabelIndex][predictedLabelIndex])
                currentAxes.text(predictedLabelIndex,trueLabelIndex,matrixElementLabelString,horizontalalignment="center",verticalalignment="center",color="k") #A numerical label for each extry in the confusion matrix is created.
        
    figure.savefig(imageSaveFilePath)