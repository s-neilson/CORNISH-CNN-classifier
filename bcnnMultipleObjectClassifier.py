import os
import numpy
from keras.models import load_model

from loadConfiguration import Configuration
from objectCreation import createImagedObjectsFromFolderOnly
from createModelPerformancePlots import createConfusionMatricies
from getObjectHierarchyLabels import getObjectHierarchyLabels
from trainBCNN import getF1ScoreOfValidationData






def main():
    #Below various configurations are loaded.
    testingConfiguration=Configuration(os.getcwd()+"/configurations/testingConfiguration.txt","=")
    
    modelFileName=testingConfiguration.getConfigurationValue("modelFileName","raw")
    allowedFileSuffixes=testingConfiguration.getConfigurationValue("useFileSuffix","raw")
    allowedFileSuffixes=[allowedFileSuffixes] if(type(allowedFileSuffixes)==str) else allowedFileSuffixes
    desiredImageSize=testingConfiguration.getConfigurationValue("desiredImageSize","int")
    dataFolder=os.getcwd()+testingConfiguration.getConfigurationValue("dataFolder","raw")
    contigiousEqualAreaRejectionCheck=testingConfiguration.getConfigurationValue("contigiousEqualAreaRejectionCheck","bool")
    contigiousEqualAreaRejectionThreshold=testingConfiguration.getConfigurationValue("contigiousEqualAreaRejectionThreshold","int") if(contigiousEqualAreaRejectionCheck) else None
    objectTypeLabels=testingConfiguration.getConfigurationValue("allowedObjectType","raw")
    
    #A map between object types and their corresponding label lists is created.
    objectTypeLabelDictionary={i[0]:tuple(i[1:]) for i in objectTypeLabels}   
    objectTypePossibleLabelSets,objectHierarchyDepth=getObjectHierarchyLabels(list(objectTypeLabelDictionary.values()))
    
    
    
    #The loaded input configuration is printed.
    print("Input testing configuration loaded:")
    print(" File name of B-CNN model to test: "+modelFileName)
    loadedModel=load_model(modelFileName)

    
    print(" Will use the following file suffixes:")
    for currentSuffix in allowedFileSuffixes:
        print("  "+currentSuffix)
        
    print(" Image size: "+str(desiredImageSize)+" pixels")
    print(" Main image folder: "+dataFolder)
    print(" Contigious colour area rejection threshold: "+(str(contigiousEqualAreaRejectionThreshold) if(contigiousEqualAreaRejectionCheck) else "Disabled"))

    
    print(" Labels at each level in the object type heirarchy:")
    for i in range(0,objectHierarchyDepth):
        print("  Level "+str(i)+":")
    
        for j in objectTypePossibleLabelSets[i]:
            print("   "+j)




    #The validation obejcts are loaded from the data folder.   
    validationObjects=createImagedObjectsFromFolderOnly(dataFolder,objectTypeLabelDictionary,desiredImageSize,contigiousEqualAreaRejectionThreshold,allowedFileSuffixes)
    
    
    #Data from the loaded ImagedObjects is turned into a format that can be used in the neural network.
    numpy.random.shuffle(validationObjects)
        
    validationObjectImageList=[currentObject.imageData for currentObject in validationObjects]
    
    
    
    
    #The above data is put into a form that can be used by the F1 scoring function.
    
    #Creates a list that contains the labels for each object represented as integers instead of strings.
    validationObjectIntegerLabelList=[[None for j in range(0,len(validationObjects))]for i in range(0,objectHierarchyDepth)]
    for i in range(0,objectHierarchyDepth):
        validationObjectIntegerLabelList[i]=[(objectTypePossibleLabelSets[i]).index(currentObject.label[i]) for currentObject in validationObjects]
     
        
    xValidation=numpy.zeros(shape=(len(validationObjectImageList),)+validationObjectImageList[0].shape)      
    for currentIndex,currentImageData in enumerate(validationObjectImageList):
        xValidation[currentIndex,:,:,:]=currentImageData


    
    #The model is tested below    
    outputConfusionMatriciesFileName="validationConfusionMatricies.png"
    print("\n")
    print("Creating confusion matricies, plot will be saved at location: "+os.getcwd()+"/"+outputConfusionMatriciesFileName)
    createConfusionMatricies(model=loadedModel,testObjects=validationObjects,testObjectImageList=validationObjectImageList,
                             imageSaveFilePath=outputConfusionMatriciesFileName,objectHierarchyLabels=objectTypePossibleLabelSets)
    print("\n")
    print("Calculating F1 score:")
    validationF1Score=getF1ScoreOfValidationData(model=loadedModel,xValidation=xValidation,validationObjectIntegerLabelList=validationObjectIntegerLabelList,
                                                 batchSize=32,objectHierarchyDepth=objectHierarchyDepth)
    print("F1 score for validation data is: "+str(validationF1Score))
    
 

main()   