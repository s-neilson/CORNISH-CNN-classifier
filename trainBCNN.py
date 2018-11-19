import keras.optimizers
import keras.callbacks
import keras.optimizers
import keras.backend as K
from hyperopt import STATUS_OK

from createParametricBCNN import createParametricBCNN




#Clears the model from GPU memory, only returns information needed for hyperparameter tuning using hyperopt.
def runOptimisingTrial(optimisingParameters,nonOptimisingModelParameters,nonOptimisingTrainParameters,nonOptimisingF1Parameters):
    model,history,f1Score=runTrial(optimisingParameters,nonOptimisingModelParameters,nonOptimisingTrainParameters,nonOptimisingF1Parameters)
        
    #The GPUs are cleared of the model so they are ready for the next one; the model must be saved beforehand to provent the weights from being reset.
    K.clear_session()
    return {"loss":-f1Score,"status":STATUS_OK}
        
    
def runTrial(optimisingParameters,nonOptimisingModelParameters,nonOptimisingTrainParameters,nonOptimisingF1Parameters):    
    dropoutFraction,convolutionLayersPerBlock,extraFirstBlock,initalLayerFilterCount,filterCountBlockMultiplicativeFactor,initalLayerKernalSize,kernalSizeBlockMultiplicitiveFactor,learningRate=optimisingParameters
    inputShape,outputLayerNames,objectTypePossibleLabelSets,gpuQuantity=nonOptimisingModelParameters
    xTrain,xValidation,yTrain,yValidation,batchSize,epochNumber,trainingLossWeight,earlyStoppingMinDelta,earlyStoppingPatience=nonOptimisingTrainParameters
    validationObjectIntegerLabelList,objectHierarchyDepth=nonOptimisingF1Parameters
      
    print("\n")
    print("Begining training run, parameters used are:")
    print(" dropoutFraction: "+str(dropoutFraction))
    print(" extraFirstBlock: "+str(extraFirstBlock))
    print(" convolutionLayersPerBlock: "+str(convolutionLayersPerBlock))
    print(" initalLayerFilterCount: "+str(initalLayerFilterCount))
    print(" filterCountBlockMultiplicativeFactor: "+str(filterCountBlockMultiplicativeFactor))
    print(" initalLayerKernalSize: "+str(initalLayerKernalSize))
    print(" kernalSizeBlockMultiplicitiveFactor: "+str(kernalSizeBlockMultiplicitiveFactor))
    print(" learningRate: "+str(learningRate))
    
    model,gpuModel=createParametricBCNN(optimisingParameters=optimisingParameters,inputShape=inputShape,outputLayerNames=outputLayerNames,
                                        objectTypePossibleLabelSets=objectTypePossibleLabelSets,gpuQuantity=gpuQuantity)
        
    print("\n")
    print(model.summary())
    history=trainBCNN(model=gpuModel,xTrain=xTrain,xValidation=xValidation,yTrain=yTrain,yValidation=yValidation,
                      batchSize=batchSize,epochsPerHierarchyLevel=epochNumber,trainingLossWeight=trainingLossWeight,
                      learningRate=learningRate,outputLayerNames=outputLayerNames,
                      earlyStoppingMinDelta=earlyStoppingMinDelta,earlyStoppingPatience=earlyStoppingPatience) 
        
    f1Score=getF1ScoreOfValidationData(model=model,xValidation=xValidation,
                                       validationObjectIntegerLabelList=validationObjectIntegerLabelList,
                                       batchSize=batchSize,objectHierarchyDepth=objectHierarchyDepth)
        
    print("Training run complete, f1 score of the validation data for this training run is "+str(f1Score)) 
    return model,history,f1Score


#Trains a B-CNN by training mainly on the loss of each output successively. outputLayerNames needs to be a list of the output layers in sequential order from
#the output closest to the input layer.
def trainBCNN(model,xTrain,xValidation,yTrain,yValidation,batchSize,epochsPerHierarchyLevel,trainingLossWeight,learningRate,outputLayerNames,earlyStoppingMinDelta,earlyStoppingPatience):
    #For each layer in the obejct hierarchy that is trainined in the CNN, that layer has a higher loss weight than the othher layers in the hierarchy.
    #This is so then training is focused on only one level of the hierarchy at a time.
    lossWeights=[]
    for currentTrainingRun in range(0,len(outputLayerNames)):
        nonTrainingLossWeight=(1.0-trainingLossWeight)/float(len(outputLayerNames)-1) #Loss weight applied to outputs not being trained on; it is done this way so trainingLossWeight corresponds to a fraction between 0 and 1.
        currentWeightValues=[trainingLossWeight if(currentIndex==currentTrainingRun) else nonTrainingLossWeight for currentIndex in range(0,len(outputLayerNames))]
        currentWeights=dict(zip(outputLayerNames,currentWeightValues))
        lossWeights.append(currentWeights)
        
       
    
    print("Training of B-CNN model will now begin, there are "+str(len(outputLayerNames))+" hierarchy levels; each of which will be trained for "+str(epochsPerHierarchyLevel)+" epochs"+"\n")
    
    modelHistory=None #Stores the history for all the training runs.
    currentOptimizer=keras.optimizers.Adam(lr=learningRate) #Done so the optimizer does not lose it's state when the model is recompiled.

    for i,currentLossWeights in enumerate(lossWeights):
        print("Training hierarchy level "+str(i+1)+", loss weights currently: "+str(list(currentLossWeights.values()))+"\n")
        model.compile(optimizer=currentOptimizer,loss="categorical_crossentropy",metrics=["accuracy"],loss_weights=currentLossWeights)
        
        earlyStoppingCallbackMonitorString="val_"+outputLayerNames[i]+"_loss" #The loss of the current model output that is being trained is monitored for early stopping.
        earlyStoppingCallback=keras.callbacks.EarlyStopping(monitor=earlyStoppingCallbackMonitorString,min_delta=earlyStoppingMinDelta,patience=earlyStoppingPatience,mode="auto",restore_best_weights=True) #Allows the model to stop training the current output if it is not improving; if so the weights will be reverted to the that the current monitored loss was at it's best value. 
        currentModelFit=model.fit(x=xTrain,y=yTrain,batch_size=batchSize,validation_data=(xValidation,yValidation),epochs=epochsPerHierarchyLevel,verbose=2,callbacks=[earlyStoppingCallback])
        print("\n"+"Training of hierarchy level "+str(i+1)+" completed")
    
    
        currentModelHistory=currentModelFit.history
    
        if(modelHistory==None):
            modelHistory=currentModelHistory #The model history for the first training run.
        else:
            #The history for this training run is added to the combined history of the previous training run/s.
            modelHistory={currentKey:modelHistory[currentKey]+currentModelHistory[currentKey] for currentKey in modelHistory.keys()}
    
    return modelHistory



def getF1ScoreOfValidationData(model,xValidation,validationObjectIntegerLabelList,batchSize,objectHierarchyDepth):
    predictedProbabilities=model.predict(x=xValidation,batch_size=batchSize)
    numberOfValidationObjects=xValidation.shape[0]
    predictedClasses=[] #Holds a list of the labels for each object.
    for i in range(0,numberOfValidationObjects): #Loops through each validation object.
        currentObjectPredictedProbabilities=[currentLevelPredictedProbabilies[i,:] for currentLevelPredictedProbabilies in predictedProbabilities]
        currentObjectPredictedClasses=[currentProbabilites.argmax() for currentProbabilites in currentObjectPredictedProbabilities] #Represents the most likely labels using integers.
        predictedClasses.append(currentObjectPredictedClasses)
        
        
    possibleClassSets=[] #A list that holds all of the possible sets of true labels. Stores data as a string to make comparisons easier.
    trueClasses=[] #Similar to predictedClasses but for the true labels of the validation objects.
    for currentObjectIndex in range(0,numberOfValidationObjects):
        currentObjectTrueClasses=[validationObjectIntegerLabelList[currentHierarchyLevel][currentObjectIndex] for currentHierarchyLevel in range(0,objectHierarchyDepth)]
        trueClasses.append(currentObjectTrueClasses)
    
        currentObjectTrueClassesString=str(currentObjectTrueClasses)
        if(not (currentObjectTrueClassesString in possibleClassSets)): #If the current object has a new unique set of object labels.
            possibleClassSets.append(currentObjectTrueClassesString)
        

        
        #Dictionaries that hold the results for every real type of object; they are uses to create f1 scores for each real type of object.
        truePositives={currentPossibleClassSet:0.0 for currentPossibleClassSet in possibleClassSets}
        falsePositives={currentPossibleClassSet:0.0 for currentPossibleClassSet in possibleClassSets}
        falseNegatives={currentPossibleClassSet:0.0 for currentPossibleClassSet in possibleClassSets}

    for currentObjectIndex in range(0,numberOfValidationObjects): #Loops through all validation objects.
        currentObjectTrueClasses=trueClasses[currentObjectIndex]
        currentObjectPredictedClasses=predictedClasses[currentObjectIndex]
    
        currentObjectTrueClassesString=str(currentObjectTrueClasses)
        currentObjectPredictedClassesString=str(currentObjectPredictedClasses)
    
        if(currentObjectTrueClassesString==currentObjectPredictedClassesString):
            truePositives[currentObjectTrueClassesString]+=1.0 #The object has been identified correctly.
        else:
            if(currentObjectPredictedClassesString in possibleClassSets): #If the predicted classes corresponds to a real type of object. 
                falsePositives[currentObjectPredictedClassesString]+=1.0 #The current object has meed misidentified as the type currentObjectPredictedClassesString.
         
            falseNegatives[currentObjectTrueClassesString]+=1.0 #The object has failed to be identified as being of the type currentObjectTrueClassesString. 

        
    resultantF1Score=0.0
    for currentClassSetString in possibleClassSets:
        currentTruePositives=truePositives[currentClassSetString]
        currentFalseNegatives=falseNegatives[currentClassSetString]
        currentFalsePositives=falsePositives[currentClassSetString]
    
        currentF1Score=(2*currentTruePositives)/((2*currentTruePositives)+currentFalsePositives+currentFalseNegatives)
        resultantF1Score+=currentF1Score/float(len(possibleClassSets)) #The total f1 score is an average of the f1 scores for each real object type.
       
    return resultantF1Score
