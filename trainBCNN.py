import os
import numpy
from keras.models import load_model
import keras.optimizers


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
        