import os
import warnings
warnings.simplefilter("ignore")
import csv
import numpy
import hyperopt
from hyperopt import Trials,tpe,hp,fmin
from keras.utils import to_categorical
import pickle

from loadConfiguration import Configuration
from objectCreation import createImagedObjects
from trainBCNN import runOptimisingTrial,runTrial
from createModelPerformancePlots import createAccuracyPlots,createConfusionMatricies
from getObjectHierarchyLabels import getObjectHierarchyLabels







def main():
    #Below various configurations are loaded.
    inputConfiguration=Configuration(os.getcwd()+"/configurations/inputConfiguration.txt","=")
    
    trainSingleModel=inputConfiguration.getConfigurationValue("trainSingleModel","bool")
    allowedFileSuffixes=inputConfiguration.getConfigurationValue("useFileSuffix","raw")
    allowedFileSuffixes=[allowedFileSuffixes] if(type(allowedFileSuffixes)==str) else allowedFileSuffixes
    desiredImageSize=inputConfiguration.getConfigurationValue("desiredImageSize","int")
    desiredImageFov=inputConfiguration.getConfigurationValue("desiredImageFov","float")
    dataFolder=os.getcwd()+inputConfiguration.getConfigurationValue("dataFolder","raw")
    contigiousEqualAreaRejectionThreshold=inputConfiguration.getConfigurationValue("contigiousEqualAreaRejectionThreshold","int")
    objectLeafLabelTotalQuantity=inputConfiguration.getConfigurationValue("objectLeafLabelTotalQuantity","int")
    transformedObjectImageRemovalChance=inputConfiguration.getConfigurationValue("transformedObjectImageRemovalChance","float")
    objectTypeLabels=inputConfiguration.getConfigurationValue("allowedObjectType","raw")
    
    #A map between object types and their corresponding label lists is created.
    objectTypeLabelDictionary={i[0]:tuple(i[1:]) for i in objectTypeLabels}   
    objectTypePossibleLabelSets,objectHierarchyDepth=getObjectHierarchyLabels(list(objectTypeLabelDictionary.values()))
    
    
    
    #The loaded input configuration is printed.
    print("Input configuration loaded:")
    print( "Will train a single model" if(trainSingleModel) else " Will optimise hyperparameters")
    
    print(" Will use the following file suffixes:")
    for currentSuffix in allowedFileSuffixes:
        print("  "+currentSuffix)
        
    print(" Image size: "+str(desiredImageSize)+" pixels")
    print(" Image field of view: "+str(desiredImageFov))
    print(" Main image folder: "+dataFolder)
    print(" Contgious colour area rejection threshold: "+str(contigiousEqualAreaRejectionThreshold))
    print(" Minimum objects per object category to load/create: "+str(objectLeafLabelTotalQuantity))
    print(" Chance of a individual image being removed from an augmented image: "+str(transformedObjectImageRemovalChance))
    
    print(" Labels at each level in the object type heirarchy:")
    for i in range(0,objectHierarchyDepth):
        print("  Level "+str(i)+":")
    
        for j in objectTypePossibleLabelSets[i]:
            print("   "+j)



    trainingConfiguration=Configuration(os.getcwd()+"/configurations/trainingConfiguration.txt","=")
    
    batchSize=trainingConfiguration.getConfigurationValue("batchSize","int")
    epochNumber=trainingConfiguration.getConfigurationValue("epochNumber","int")
    trainingLossWeight=trainingConfiguration.getConfigurationValue("trainingLossWeight","float")
    validationFraction=trainingConfiguration.getConfigurationValue("validationFraction","float")   
    outputFilePrefix=trainingConfiguration.getConfigurationValue("outputFilePrefix","raw")
    
    
    hyperparameterOptimisationMaximumEvaluations=trainingConfiguration.getConfigurationValue("hyperparameterOptimisationMaximumEvaluations","int")
    dropoutFraction=trainingConfiguration.getConfigurationValue("dropoutFraction","float")
    convolutionLayersPerBlock=trainingConfiguration.getConfigurationValue("convolutionLayersPerBlock","int")
    extraFirstBlock=trainingConfiguration.getConfigurationValue("extraFirstBlock","bool")
    initalLayerFilterCount=trainingConfiguration.getConfigurationValue("initalLayerFilterCount","int")
    filterCountBlockMultiplicativeFactor=trainingConfiguration.getConfigurationValue("filterCountBlockMultiplicativeFactor","float")
    initalLayerKernalSize=trainingConfiguration.getConfigurationValue("initalLayerKernalSize","int")
    kernalSizeBlockMultiplicitiveFactor=trainingConfiguration.getConfigurationValue("kernalSizeBlockMultiplicitiveFactor","float")
    learningRate=trainingConfiguration.getConfigurationValue("learningRate","float")
    
    gpuQuantity=trainingConfiguration.getConfigurationValue("gpuQuantity","int")
    earlyStoppingMinDelta=trainingConfiguration.getConfigurationValue("earlyStoppingMinDelta","float")
    earlyStoppingPatience=trainingConfiguration.getConfigurationValue("earlyStoppingPatience","int")
    
    
    #the loaded training configuration is printed.
    print("\n")
    print("Training configuration loaded:")
    print(" Batch size: "+str(batchSize))
    print(" Epochs trained per level in hierarchy: "+str(epochNumber))
    print(" Current hierarchy level training loss weight: "+str(trainingLossWeight))
    print(" Validation object fraction: "+str(validationFraction))
    print(" Output file prefix: "+outputFilePrefix)
    
    if(trainSingleModel):     
        print(" The following parameters will be used for training the model:")
        print("  dropoutFraction: "+str(dropoutFraction))
        print("  convolutionLayersPerBlock: "+str(convolutionLayersPerBlock))
        print("  extraFirstBlock: "+str(extraFirstBlock))
        print("  initalLayerFilterCount: "+str(initalLayerFilterCount))
        print("  filterCountBlockMultiplicativeFactor: "+str(filterCountBlockMultiplicativeFactor))
        print("  initalLayerKernalSize: "+str(initalLayerKernalSize))
        print("  kernalSizeBlockMultiplicitiveFactor: "+str(kernalSizeBlockMultiplicitiveFactor))
        print("  learningRate: "+str(learningRate))
    else:  
        print(" Maximum number of hyperparameter optimisation evaluations: "+str(hyperparameterOptimisationMaximumEvaluations))

    print(" Number of GPUs to use for training: "+str(gpuQuantity))
    print(" Early stopping minimum loss delta: "+str(earlyStoppingMinDelta))
    print(" Early stopping patience: "+str(earlyStoppingPatience))
    
    
    
    
    hyperparameterLimitsConfiguration=Configuration(os.getcwd()+"/configurations/hyperparameterLimitsConfiguration.txt","=")
    
    minimumDropoutFraction=hyperparameterLimitsConfiguration.getConfigurationValue("minimumDropoutFraction","float")
    maximumDropoutFraction=hyperparameterLimitsConfiguration.getConfigurationValue("maximumDropoutFraction","float")
    possibleConvolutionLayersPerBlock=hyperparameterLimitsConfiguration.getConfigurationValue("possibleConvolutionLayersPerBlock","int")
    possibleInitalLayerFilterCount=hyperparameterLimitsConfiguration.getConfigurationValue("possibleInitalLayerFilterCount","int")
    possibleFilterCountBlockMultiplicativeFactor=hyperparameterLimitsConfiguration.getConfigurationValue("possibleFilterCountBlockMultiplicativeFactor","float")
    possibleInitalLayerKernalSize=hyperparameterLimitsConfiguration.getConfigurationValue("possibleInitalLayerKernalSize","int")
    possibleKernalSizeBlockMultiplicitiveFactor=hyperparameterLimitsConfiguration.getConfigurationValue("possibleKernalSizeBlockMultiplicitiveFactor","float")
    minimumLearningRate=hyperparameterLimitsConfiguration.getConfigurationValue("minimumLearningRate","float")
    maximumLearningRate=hyperparameterLimitsConfiguration.getConfigurationValue("maximumLearningRate","float")
    
    
    if(not trainSingleModel):
        print("\n")
        print(" Hyperparameters will be optimised through the following ranges:")
        print("  dropoutFraction: "+str(minimumDropoutFraction)+"-"+str(maximumDropoutFraction))
        print("  convolutionLayersPerBlock: "+str(possibleConvolutionLayersPerBlock))
        print("  extraFirstBlock: True or False")
        print("  initalLayerFilterCount: "+str(possibleInitalLayerFilterCount))
        print("  filterCountBlockMultiplicativeFactor: "+str(possibleFilterCountBlockMultiplicativeFactor))
        print("  initalLayerKernalSize: "+str(possibleInitalLayerKernalSize))
        print("  kernalSizeBlockMultiplicitiveFactor: "+str(possibleKernalSizeBlockMultiplicitiveFactor))
        print("  learningRate: "+str(minimumLearningRate)+"-"+str(maximumLearningRate))
    
    
    
    
    
    
    
    trainObjects,validationObjects=createImagedObjects(dataFolder,objectTypeLabelDictionary,desiredImageSize,desiredImageFov,
                                                       contigiousEqualAreaRejectionThreshold,allowedFileSuffixes,validationFraction,  
                                                       objectLeafLabelTotalQuantity,transformedObjectImageRemovalChance)
    
    
    #Data from the loaded/created ImagedObjects is turned into a format that can be used in the neural network.
    numpy.random.shuffle(trainObjects)
    numpy.random.shuffle(validationObjects)
    
    trainObjectImageList=[currentObject.imageData for currentObject in trainObjects]
    trainObjectIntegerLabelList=[[None for j in range(0,len(trainObjects))]for i in range(0,objectHierarchyDepth)]
    
    validationObjectImageList=[currentObject.imageData for currentObject in validationObjects]
    validationObjectIntegerLabelList=[[None for j in range(0,len(validationObjects))]for i in range(0,objectHierarchyDepth)]
    
    #Creates a list that contains the labels for each object represented as integers instead of strings.
    for i in range(0,objectHierarchyDepth):
        trainObjectIntegerLabelList[i]=[(objectTypePossibleLabelSets[i]).index(currentObject.label[i]) for currentObject in trainObjects]
        validationObjectIntegerLabelList[i]=[(objectTypePossibleLabelSets[i]).index(currentObject.label[i]) for currentObject in validationObjects]
    
    #The labels are one-hot encoded for each level in the object heirarchy.
    trainLabels=[to_categorical(trainObjectIntegerLabelList[i],len(objectTypePossibleLabelSets[i])) for i in range(0,objectHierarchyDepth)]
    validationLabels=[to_categorical(validationObjectIntegerLabelList[i],len(objectTypePossibleLabelSets[i])) for i in range(0,objectHierarchyDepth)]
    
    
    
    #The above data is put into a form that can be used by the model.
    xTrain=numpy.zeros(shape=(len(trainObjectImageList),)+trainObjectImageList[0].shape)
    xValidation=numpy.zeros(shape=(len(validationObjectImageList),)+validationObjectImageList[0].shape)
    
    for currentIndex,currentImageData in enumerate(trainObjectImageList):
        xTrain[currentIndex,:,:,:]=currentImageData
        
    for currentIndex,currentImageData in enumerate(validationObjectImageList):
        xValidation[currentIndex,:,:,:]=currentImageData
    
    #Each output of the model is accociated with a set of labels.
    outputLayerNames=["out"+str(i+1) for i in range(0,objectHierarchyDepth)] #Each output layer is labeled sequentially from the output closest to the input layer.
    yTrain=dict(zip(outputLayerNames,trainLabels))
    yValidation=dict(zip(outputLayerNames,validationLabels))
    
  
    
    
    
    nonOptimisingModelParameters=validationObjectImageList[0].shape,outputLayerNames,objectTypePossibleLabelSets,gpuQuantity
    nonOptimisingTrainParameters=xTrain,xValidation,yTrain,yValidation,batchSize,epochNumber,trainingLossWeight,earlyStoppingMinDelta,earlyStoppingPatience
    nonOptimisingF1Parameters=validationObjectIntegerLabelList,objectHierarchyDepth
    
    if(trainSingleModel): #For training a single model with specified hyperparameters.
        modelHyperparameters=[dropoutFraction,convolutionLayersPerBlock,extraFirstBlock,initalLayerFilterCount,filterCountBlockMultiplicativeFactor,initalLayerKernalSize,kernalSizeBlockMultiplicitiveFactor,learningRate]
        result=runTrial(modelHyperparameters,nonOptimisingModelParameters,nonOptimisingTrainParameters,nonOptimisingF1Parameters)
        
        outputModelFileName=outputFilePrefix+"TrainedModel.h5"
        outputModelHistoryFileName=outputFilePrefix+"TrainingHistory.npy"
        print("\n")
        print("Saving model file at location "+os.getcwd()+"/"+outputModelFileName)
        print("Saving model training history file at location "+os.getcwd()+"/"+outputModelHistoryFileName)
        result[0].save(outputModelFileName) #The model is saved.
        numpy.save(outputModelHistoryFileName,result[1]) #The training history is saved.
        
        outputConfusionMatriciesFilePath=outputFilePrefix+"ConfusionMatricies.png"
        print("\n")
        print("Creating accuracy plots, will be saved in folder "+os.getcwd()+" as .png files with the prefix "+outputFilePrefix+"AccuracyPlot_")
        createAccuracyPlots(result[1],outputLayerNames,outputFilePrefix)
        print("Creating confusion matricies, plot will be saved at location: "+os.getcwd()+"/"+outputConfusionMatriciesFilePath)
        createConfusionMatricies(model=result[0],testObjects=validationObjects,testObjectImageList=validationObjectImageList,
                                 imageSaveFilePath=outputConfusionMatriciesFilePath,objectHierarchyLabels=objectTypePossibleLabelSets)
    else: #For hyperparameter optimisation    
        rotLambda=lambda parameters:runOptimisingTrial(parameters,nonOptimisingModelParameters,nonOptimisingTrainParameters,nonOptimisingF1Parameters)
        
        space=[hp.uniform("dropoutFraction",minimumDropoutFraction,maximumDropoutFraction),
               hp.choice("convolutionLayersPerBlock",possibleConvolutionLayersPerBlock),
               hp.choice("extraFirstBlock",[True,False]),
               hp.choice("initalLayerFilterCount",possibleInitalLayerFilterCount),
               hp.choice("filterCountBlockMultiplicativeFactor",possibleFilterCountBlockMultiplicativeFactor),
               hp.choice("initalLayerKernalSize",possibleInitalLayerKernalSize),
               hp.choice("kernalSizeBlockMultiplicitiveFactor",possibleKernalSizeBlockMultiplicitiveFactor),
               hp.uniform("learningRate",minimumLearningRate,maximumLearningRate)]
    
        trials=Trials()
        bestResults=fmin(rotLambda,space=space,algo=tpe.suggest,max_evals=hyperparameterOptimisationMaximumEvaluations,trials=trials)
        
    
        
        optimisedHyperparameters=hyperopt.space_eval(space,bestResults)
        print("\n")
        print("Optimised hyperparameters: ",optimisedHyperparameters)
        outputOptimisedHyperparameterFileName=outputFilePrefix+"OptimisedHyperparameters.txt"
        print("Saving optimised hyperparmeters at location: "+os.getcwd()+"/"+outputOptimisedHyperparameterFileName)
        outputOptimisedHyperparameterFile=open(outputOptimisedHyperparameterFileName,"w")
        
        outputOptimisedHyperparameterFileWriter=csv.writer(outputOptimisedHyperparameterFile,delimiter="=")
        outputOptimisedHyperparameterFileWriter.writerow(["dropoutFraction",optimisedHyperparameters[0]])
        outputOptimisedHyperparameterFileWriter.writerow(["convolutionLayersPerBlock",optimisedHyperparameters[1]])
        outputOptimisedHyperparameterFileWriter.writerow(["extraFirstBlock",optimisedHyperparameters[2]])
        outputOptimisedHyperparameterFileWriter.writerow(["initalLayerFilterCount",optimisedHyperparameters[3]])
        outputOptimisedHyperparameterFileWriter.writerow(["filterCountBlockMultiplicativeFactor",optimisedHyperparameters[4]])
        outputOptimisedHyperparameterFileWriter.writerow(["initalLayerKernalSize",optimisedHyperparameters[5]])
        outputOptimisedHyperparameterFileWriter.writerow(["kernalSizeBlockMultiplicitiveFactor",optimisedHyperparameters[6]])
        outputOptimisedHyperparameterFileWriter.writerow(["learningRate",optimisedHyperparameters[7]])
        
        outputOptimisedHyperparameterFile.close()
        
        print("\n")
        outputTrialsFileName=outputFilePrefix+"TrainingTrials.p"
        print("Saving trials pickle file at location "+os.getcwd()+"/"+outputTrialsFileName)       
        pickle.dump(trials,open(outputTrialsFileName,"wb"))
 

main()



    