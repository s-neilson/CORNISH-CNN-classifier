import numpy

#A generator that creates batches of object images and their labels as they are needed.
def batchDataGenerator(batchSize,imageList,objectLabels,networkOutputNames):
    currentBatchImages=numpy.zeros(shape=(batchSize,)+imageList[0].shape)
    currentBatchLabels=[numpy.zeros(shape=(batchSize,objectLabels[i].shape[1])) for i in range(0,len(objectLabels))] 
    #There is a list for each level in the object hierarchy, with each of these lists containing the appropiate
    #labels for each object; labels which are lists themselves in ont-hot encoded form.
    
    currentBatchCount=0
    currentObjectIndex=0   
    numberOfObjects=len(imageList)
    #lop=[None for i in range(0,batchSize)]
    
    while True: #The generator will create batches of data continously.
        
        for i in range(0,batchSize): #A loop that adds the data for the next batchSize number of objects to the output variables.
            if(currentObjectIndex==numberOfObjects):
                currentObjectIndex=0 #The object index is looped back to the beginning if the last object has been reached.
            
            currentBatchImages[currentBatchCount,:,:,:]=imageList[currentObjectIndex] #Image data is added.
            #lop[currentBatchCount]=currentObjectIndex
        
            #Extracts the labels for a current object and places references to them in the output label list. 
            for currentLevelIndex in range(0,len(objectLabels)): #A loop for each level in the object hierarchy
                currentBatchLabels[currentLevelIndex][currentBatchCount,:]=objectLabels[currentLevelIndex][currentObjectIndex,:] 

            currentBatchCount+=1
            currentObjectIndex+=1

        #Labels are in dictionary form so each label array level is associated with a specific output layer of the network.
        outputLabelDictionary=dict(zip(networkOutputNames,currentBatchLabels))
        yield (currentBatchImages,outputLabelDictionary) 

        currentBatchCount=0 #Number of objects currently in batch is reset to zero in order to be ready for the next batch        