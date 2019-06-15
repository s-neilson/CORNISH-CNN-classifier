import os
import numpy
from tqdm import tqdm,trange
from sklearn.model_selection import train_test_split

from imagedObject import FolderImagedObject,TransformedImagedObject





def loadImagedObjects(mainObjectFolder,objectTypeLabelDictionary,imageSize,rejectionThresholdArea,fileSuffixes):
    loadedObjects={currentLabels[-1]:[] for currentLabels in objectTypeLabelDictionary.values()} #Holds the loaded objects associated with the last component of their label set (their leaf label).
    objectsToLoadData=[] #Holds dictionaries that contain information about the objects to be loaded
    
    
    for currentObjectType in objectTypeLabelDictionary.keys():
        currentTypeFolder=mainObjectFolder+"/"+currentObjectType
        currentTypeFolderContents=os.listdir(currentTypeFolder)
        
        #Gets all the objects of a certain folder type.
        for currentContentName in currentTypeFolderContents:
            currentObjectFolder=currentTypeFolder+"/"+currentContentName
            
            if(not os.path.isfile(currentObjectFolder)): #If it is a folder
                currentObjectLabel=objectTypeLabelDictionary[currentObjectType] #The correct label is obtained for this object type.
                currentObjectDataDictionary={"name":currentContentName,"label":currentObjectLabel,"folder":currentObjectFolder}
                objectsToLoadData.append(currentObjectDataDictionary)
    
    print("\n")        
    print(str(len(objectsToLoadData))+" objects will be imported")
    for currentObjectImportData in tqdm(objectsToLoadData):
        newPath=currentObjectImportData["folder"]
        newName=currentObjectImportData["name"]
        newLabel=currentObjectImportData["label"]
        newObject=FolderImagedObject(newPath,newName,newLabel,imageSize,rejectionThresholdArea,fileSuffixes)
        
        if(newObject.nonBlankImageCount>0): #If at least one image was succesfully loaded for the object.
            leafLabel=newLabel[-1] #The final label for the object.
            loadedObjects[leafLabel].append(newObject) #The loaded object is added to the list associated with it's leaf label.
        
    print("Importing of objects complete, loaded object leaf label counts are:")
    print("\n")
    for currentObjectType,currentObjectList in loadedObjects.items():
        print(currentObjectType+": "+str(len(currentObjectList)))

    return loadedObjects



def createTransformedImagedObjectsForObjectType(objectList,objectLeafLabel,objectLeafLabelQuantityLimit,imageRemovalChance):
    print("\n")
    print("There are currently "+str(len(objectList))+" objects with the leaf label "+objectLeafLabel)
    
    if(len(objectList)>=objectLeafLabelQuantityLimit):
        print("No transformed objects with leaf label "+objectLeafLabel+" will be created as the current number of objects with that leaf label exceeds the object leaf label quantity limit")
        return [] #An empty list of new objects is returned.
    else:
        objectQuantityToCreate=objectLeafLabelQuantityLimit-len(objectList)
        print(str(objectQuantityToCreate)+" transformed objects will be created for objects with leaf label "+objectLeafLabel)
        createdTransformedObjects=[]
        
        for i in trange(0,objectQuantityToCreate):
            transformedObjectTemplateObject=objectList[numpy.random.randint(0,len(objectList))] #A random object is chosen as the template for the transformed object.
            newObject=TransformedImagedObject(transformedObjectTemplateObject,imageRemovalChance)
            createdTransformedObjects.append(newObject)
            
    return createdTransformedObjects
    



def createImagedObjects(importFolderPath,objectTypeLabelDictionary,imageSize,rejectionThresholdArea,fileSuffixes,validationFraction,objectLeafLabelQuantityLimit,transformedObjectImageRemovalChance): 
    loadedObjects=loadImagedObjects(importFolderPath,objectTypeLabelDictionary,imageSize,rejectionThresholdArea,fileSuffixes)
    loadedTrainObjects={currentLeafLabel:[] for currentLeafLabel in loadedObjects.keys()} #Holds the loaded training objects associated with each type leaf label.
    validationObjects=[]
    
    print("\n")
    print("Splitting validation objects from training objects:")
    for currentObjectLeafLabel,currentObjectList in loadedObjects.items(): #Validation objects are split from the training objects.
        currentTrainObjects,currentValidationObjects=train_test_split(currentObjectList,test_size=validationFraction)
        print("Objects with leaf label "+currentObjectLeafLabel+" split into "+str(len(currentTrainObjects))+" training objects and "+str(len(currentValidationObjects))+" validation objects.")
        
        validationObjects+=currentValidationObjects #The newly split validation objects are added to the main validation object list.
        loadedTrainObjects[currentObjectLeafLabel]=currentTrainObjects #The newly split training objects are added to the loadedTrainObjects dictionary.
    print("Split of validation objects complete, there are "+str(len(validationObjects))+" validation objects in total.")
     
    trainObjects=[]
    print("\n")
    print("Each set of objects with a particular leaf label will be augmented with transformed objects in order for each object set of a particular leaf label to contain "+str(objectLeafLabelQuantityLimit)+" objects. Sets of objects with the same leaf label that are at or exceed this quantity will not be augmented.")
    for currentObjectLeafLabel,currentObjectList in loadedTrainObjects.items(): #Transformed obejcts for each category are now created.
        currentTransformedObjects=createTransformedImagedObjectsForObjectType(currentObjectList,currentObjectLeafLabel,objectLeafLabelQuantityLimit,transformedObjectImageRemovalChance)
        trainObjects+=currentObjectList #The unmodified loaded objects are added the the training object list.
        trainObjects+=currentTransformedObjects #The newly transformed objects are added to the training object list.
    print("Creation of transformed objects complete.")
     
    print("\n")
    print("There are "+str(len(trainObjects))+" training objects in total")
    
    
    return trainObjects,validationObjects



#Similar to createImagedObjects, but does no training or validaiton splitting and creates no extra transformed objects.
def createImagedObjectsFromFolderOnly(importFolderPath,objectTypeLabelDictionary,imageSize,rejectionThresholdArea,fileSuffixes):
    loadedObjects=loadImagedObjects(importFolderPath,objectTypeLabelDictionary,imageSize,rejectionThresholdArea,fileSuffixes)
    outputObjects=[] #Holds the list of all ImagedObjects.
    
    for currentObjectLeafLabel,currentObjectList in loadedObjects.items(): #Loops through all leaf object types
        outputObjects+=currentObjectList #The object list for the current leaf object type is added to the outputObjects list.
        
    return outputObjects
    