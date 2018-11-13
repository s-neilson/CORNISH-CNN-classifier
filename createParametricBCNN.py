import math
import tensorflow as tf
from keras.utils import multi_gpu_model
from keras.models import Model
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.engine.input_layer import Input


#Creates a BCNN based upon various comfiguration variables
def createParametricBCNN(dropoutFraction,convolutionLayersPerBlock,extraFirstBlock,initalLayerFilterCount,
                         filterCountBlockMultiplicativeFactor,initalLayerKernalSize,kernalSizeBlockMultiplicitiveFactor,
                         inputShape,outputLayerNames,objectTypePossibleLabelSets,gpuQuantity):
    
    #extraFirstBlock: Whether an extra block is to be included before the first output.
    #filterCountBlockMultiplicativeFactor: The amount of filters the convolution layers in a block have compared to the previous block.
    #kernalSizeBlockMultiplicitiveFactor: #The kernal size for convolutional layers in a block compared to the previous block.
    currentFilterCount=initalLayerFilterCount
    currentKernalSize=initalLayerKernalSize
    
    mainInput=Input(shape=inputShape,name="input")
    network=mainInput
    outputs=[None for i in range(0,len(outputLayerNames))]
    
    if(extraFirstBlock): 
        for i in range(0,convolutionLayersPerBlock): #Creates the convolution layers in the first block.
            network=Conv2D(currentFilterCount,(currentKernalSize,currentKernalSize),padding="same",activation="relu")(network) 
            
        network=MaxPooling2D(pool_size=(2,2))(network)
        network=Dropout(dropoutFraction)(network)
        currentFilterCount*=filterCountBlockMultiplicativeFactor
        currentKernalSize*=kernalSizeBlockMultiplicitiveFactor
        
    #The rest of the network is made below.
    for currentIndex,currentOutputName in enumerate(outputLayerNames): #Loops over all outputs; each output is associated with a block.
        for i in range(0,convolutionLayersPerBlock): #Creates the convolution layers in the block
            currentIntegerFilterCount=int(currentFilterCount)
            
            currentIntegerKernalSize=math.floor(currentKernalSize)
            currentIntegerKernalSize=currentIntegerKernalSize+1 if(currentIntegerKernalSize%2==0) else currentIntegerKernalSize #Makes sure that the kernal size is an odd number.
            currentIntegerKernalSize=max(3,currentIntegerKernalSize) #Makes sure that the kernal size is no smaller than 3.
            
            network=Conv2D(currentIntegerFilterCount,(currentIntegerKernalSize,currentIntegerKernalSize),padding="same",activation="relu")(network)
        
        #The output layers for this block are created.
        outputs[currentIndex]=Conv2D(len(objectTypePossibleLabelSets[currentIndex]),(3,3),padding="same",activation="relu",name=currentOutputName+"LocationHeatmap")(network)
        outputs[currentIndex]=GlobalAveragePooling2D()(outputs[currentIndex])
        outputs[currentIndex]=Activation("softmax",name=currentOutputName)(outputs[currentIndex])
        
        if((currentIndex+1)<len(outputLayerNames)): #If there are still blocks left to create.
            network=MaxPooling2D(pool_size=(2,2))(network)
            network=Dropout(dropoutFraction)(network)
            currentFilterCount*=filterCountBlockMultiplicativeFactor
            currentKernalSize*=kernalSizeBlockMultiplicitiveFactor    
    
    
    #Initally creates the model on a CPU instead of a GPU.        
    with tf.device("/cpu:0"):
        model=Model(inputs=mainInput,outputs=outputs)   
        
    gpuModel=multi_gpu_model(model,gpus=gpuQuantity)
    return model,gpuModel