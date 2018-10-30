import os
import math
import numpy
from keras.models import load_model
import keras.optimizers
from matplotlib import pyplot as plt

from loadConfiguration import Configuration

#Trains a B-CNN by training mainly on the loss of each output successively. outputLayerNames needs to be a list of the output layers in sequential order from
#the output closest to the input layer.
def trainBCNN(modelFileName,modelHistoryFileName,trainingDataGenerator,validationDataGenerator,trainingStepsPerEpoch,validationStepsPerEpoch,epochsPerHierarchyLevel,outputLayerNames,trainingLossWeight):
    #For each layer in the obejct hierarchy that is trainined in the CNN, that layer has a higher loss weight than the othher layers in the hierarchy.
    #This is so then training is focused on only one level of the hierarchy at a time.
    lossWeights=[]
    for currentTrainingRun in range(0,len(outputLayerNames)):
        nonTrainingLossWeight=(1.0-trainingLossWeight)/float(len(outputLayerNames)-1) #Loss weight applied to outputs not being trained on; it is done this way so trainingLossWeight corresponds to a fraction between 0 and 1.
        currentWeightValues=[trainingLossWeight if(currentIndex==currentTrainingRun) else nonTrainingLossWeight for currentIndex in range(0,len(outputLayerNames))]
        currentWeights=dict(zip(outputLayerNames,currentWeightValues))
        lossWeights.append(currentWeights)
        
       
    
    print("Training of B-CNN model will now begin, there are "+str(len(outputLayerNames))+" hierarchy levels; each of which will be trained for "+str(epochsPerHierarchyLevel)+"\n")
    
    modelHistory=None #Stores the history for all the training runs.

    for i,currentLossWeights in enumerate(lossWeights):
        print("Training hierarchy level "+str(i+1)+", loss weights currently: "+str(list(currentLossWeights.values()))+"\n")
        loadedModel=load_model(modelFileName)
        loadedModel.compile(optimizer=keras.optimizers.Adadelta(),loss="categorical_crossentropy",metrics=["accuracy"],loss_weights=currentLossWeights)
        
        currentModelFit=loadedModel.fit_generator(generator=trainingDataGenerator,steps_per_epoch=trainingStepsPerEpoch,validation_data=validationDataGenerator,validation_steps=validationStepsPerEpoch,epochs=epochsPerHierarchyLevel,verbose=2)
        loadedModel.save(modelFileName)
        print("\n"+"Training of hierarchy level "+str(i+1)+" completed, model saved to: "+os.getcwd()+"/"+modelFileName)
    
    
        currentModelHistory=currentModelFit.history
    
        if(modelHistory==None):
            modelHistory=currentModelHistory #The model history for the first training run.
        else:
            #The history for this training run is added to the combined history of the previous training run/s.
            modelHistory={currentKey:modelHistory[currentKey]+currentModelHistory[currentKey] for currentKey in modelHistory.keys()}
        
        numpy.save(modelHistoryFileName,modelHistory)
        
        
def createConfusionMatricies(modelFileName,modelConfigurationFileName,imageSaveFilePath,testObjects,testObjectImageList):
    loadedModel=load_model(modelFileName)   
    modelConfiguration=Configuration(modelConfigurationFileName,"=")
    objectHierarchyLabels=modelConfiguration.getConfigurationValue("labelsForHierarchyLevel","raw")
    
    numberOfValidationObjects=len(testObjectImageList)
    #testObjectImageList is turned into a numpy array with the first axis indexing each object.
    validationImageDataArray=numpy.empty(shape=(numberOfValidationObjects,)+testObjectImageList[0].shape)
    for i,currentImageData in enumerate(testObjectImageList):
        validationImageDataArray[i,:,:,:]=currentImageData


    predictedProbabilities=loadedModel.predict(x=validationImageDataArray,batch_size=32)


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
                 
        
        #Below plots the confusion matricies.  
        currentAxes=figure.add_subplot(subplotDivision,subplotDivision,i+1) 
        plt.matshow(currentConfusionMatrixFraction,cmap="winter",fignum=False)         

        currentAxes.set_xticks(range(0,currentHierarchyLevelLabelCount)) #Done to prevent labels from being assined negative positions on the plot (which is what happens by default).
        currentAxes.set_yticks(range(0,currentHierarchyLevelLabelCount)) #Done to prevent labels from being assined negative positions on the plot (which is what happens by default).
        currentAxes.set_xticklabels(objectHierarchyLabels[i])
        currentAxes.set_yticklabels(objectHierarchyLabels[i])
        currentAxes.xaxis.set_ticks_position("bottom") #Done so the horizontal axis labels don't interfere with the title.
        currentAxes.set_xlabel("True label")
        currentAxes.set_ylabel("Predicted label")
        currentAxes.set_title("Hierarchy level "+str(i+1))                     
        
        for trueLabelIndex in range(0,currentHierarchyLevelLabelCount):
            for predictedLabelIndex in range(0,currentHierarchyLevelLabelCount):
                matrixElementLabelPercentage=str(round(currentConfusionMatrixFraction[trueLabelIndex][predictedLabelIndex]*100.0,1))+"%"
                matrixElementLabelString=matrixElementLabelPercentage+"\n"+str(currentConfusionMatrix[trueLabelIndex][predictedLabelIndex])
                currentAxes.text(trueLabelIndex,predictedLabelIndex,matrixElementLabelString,horizontalalignment="center",verticalalignment="center",color="k") #A numerical label for each extry in the confusion matrix is created.
        
    figure.show()
    figure.savefig(imageSaveFilePath)
