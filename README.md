# CORNISH-CNN-classifier
A convolutional neural network for the classification of astronomical obejcts using data that is part of the CORNISH (Co-Ordinated Radio 'N' Infrared Survey for High-mass star formation) project (website: http://cornish.leeds.ac.uk/public/index.php).

## List of programs included
* fileDownloader.py: Downloads CORNISH .fits files from the CORNISH catalog (avaliable at http://cornish.leeds.ac.uk/public/catalogue.php).
* CORNISH_B-CNN_Trainer_And_Optimiser.py: Either optimises the hyperparameters of a B-CNN or trains a single B-CNN according to specified hyperparameters.
* bcnnSingleObjectClassifier.py: Classifies a single object given the a model and the .fits files that make up the object. Also generates a location heatmap for the classified object.
* bcnnMultipleObjectClassifier.py: Classifies multiple objects given a model, a data folder and information regarding the structure of .fits files in the data folder. Outputs a confusion matrix for the tested obejcts and the model's F1 score on the data.
                                                                                                                                          
## Configuration files

All file paths are relative to the location of the file downloader. Configuration names and their values are seperated by an equal sign. For configuration parameters that take multiple values each values is seperated by an equals sign like the configuration name and first value. Some configuration names can occur multiple times in a configuration file. Boolean data types interpret as string "yes" as true, all other strings are interpreted as false.

* downloadConfiguration.txt: Configuration for file downloader
* inputConfiguration.txt: Configuration for data preproprocesing when training a model/optimising hyperparameters.
* trainingConfiguration.txt: Configuration for the training of a model (or models when optimising hyperparameters).
* testingConfiguration.txt: Configuration for the multiple object testing of a model.
* hyperparameterLimitsConfiguration: Configuration containing the ranges in which hyperparameters can be optimised. The hyperparameter _extraFirstBlock_ always has the range of true and false.


### File downloader configuration (downloadConfiguration.txt)

|Name|Can have multiple values|Can occur multiple times|Description|Data type|
|----|------------------------|------------------------|-----------|---------|
|objectTypeFilePath|No|No|Path to CSV file contains pairs of object names and their associated type.|String|
|outputFileLocation|No|No|Path to the folder containing all of the downloaded files; this folder will have subfolders for every object type.|String|
|fileSourceURLPrefix|No|No|URL prefix for downloaded files; full download URL contains the prefix followed by the object name followed by a suffix depending on the specific file associated with each object.|String
|downloadPartialObjects|Whether if as many files for each object will be downloaded as possible (yes) or if an object will be skipped if one or more of it's files cannot be downloaded (otherwise).|Boolean|
|downloadFileSuffix|No|Yes|The suffixes of files that are to be downloaded.|String|

An example of a file URL to download with a fileSourceURLPrefix of http://cornish.leeds.ac.uk/public/data_src/, a name of G009.9702-00.5292 and a downloadFileSuffix of "_CORNISH_5GHz.fits" is http://cornish.leeds.ac.uk/public/data_src/G009.9702-00.5292_CORNISH_5GHz.fits


### Training input configuration (inputConfiguration.txt)

|Name|Can have multiple values|Can occur multiple times|Description|Data type|
|----|------------------------|------------------------|-----------|---------|
|trainSingleModel|No|No|Whether a single model should be trained (yes) or hyperparameters optimised (otherwise).|Boolean|
|desiredImageSize|No|No|Square edge size in pixels that input images should be scaled to.|Integer|
|contigiousEqualAreaRejectionThreshold|No|No|Images that have a contigious area of the same pixel value connected to the image edge will be rejected if the areforementioned area equals or exceeds this value. This is done to remove images that may come from the edge of an astronomical survey; such images may cause problems with training.|Integer|
|objectLeafLabelTotalQuantity|No|No|The maximum amount of imaged objects that will exist for each object type. If the amount of loaded objects of a particular object type does not reach this limit, extra objects will be created from the loaded ones using the technique of data augmentation. No new objects will be created if the amount of loaded objects for a particular object type equals or exceeds this limit.|Integer|
|transformedObjectImageRemovalChance|No|No|Chance of an individual image of an augmented (or transformed) object to be replaced with a blank image. This is done so the classifier is forced not to rely on the presence of certain image channels.|Float|
|dataFolder|No|No|Path to folder containing the object type subfolders.|String|
|filePrefix|No|No|Specifies a file prefix for images that can be used in training. Without this, the name of the object will be used|String|
|useFileSuffix|No|Yes|Specifies a file suffix (file name without the object name component) for images that can be used in training.|String|
|allowedObjectType|Yes|Yes|Ther first value specifies the object type based on object type folders that are created using the file downloader, while the remaining values specify in order the label hierarchy for this object type.|String|

### Testing input configuration (testingConfiguration.txt)

|Name|Can have multiple values|Can occur multiple times|Description|Data type|
|----|------------------------|------------------------|-----------|---------|
|modelFileName|No|No|The file path of the saved model to test relative to the multiple objects classifier .py file that is being run.|String|
|desiredImageSize|No|No|Square edge size in pixels that input images should be scaled to.|Integer|
|contigiousEqualAreaRejectionThreshold|No|No|Images that have a contigious area of the same pixel value connected to the image edge will be rejected if the areforementioned area equals or exceeds this value. This is done to remove images that may come from the edge of an astronomical survey; such images may cause problems with training.|Integer|
|dataFolder|No|No|Path to folder containing the object type subfolders.|String|
|filePrefix|No|No|Specifies a file prefix for images that can be used in training. Without this, the name of the object will be used|String|
|useFileSuffix|No|Yes|Specifies a file suffix (file name without the object name component) for images that can be used in training.|String|
|allowedObjectType|Yes|Yes|Ther first value specifies the object type based on object type folders that are created using the file downloader, while the remaining values specify in order the label hierarchy for this object type.|String|

### Training process configuration (trainingConfiguration.txt)

A "block" in the B-CNN contains a number of 2D convolutional layers with the same kernal size and number of filters that if followed by another block is attached to a 2D Maximum pooling layer (with a kernal size of 2x2) followed by a dropout layer. 

If the block contains an output, a convolutional layer with a 3x3 kernal size and quantity of filters equal to the number of possible classification labels associated with the current output is attached to the previous convolutional layer. This is then followed by a global average pooling layer and then the output layer, with is a softmax activation layer.

Early stopping is when the training process for the current output is stopped before reaching an _epochNumber_ number of epochs due to the validation loss for the current output that is being trained (the monitored loss) not improving over a specified number of epochs.


|Name|Can have multiple values|Can occur multiple times|Description|Data type|
|----|------------------------|------------------------|-----------|---------|
|batchSize|No|No|The batch size to be used while training the model/s|Integer|
|epochNumber|No|No|The maximum amount of epochs (full iterations through all training data) to train a specific output of the B-CNN until either continuing on using the next output or finishing.|Integer|
|trainingLossWeight|No|No|The total training and validation losses are weighted by this nfraction for the output that is currently being trained. The remaining fraction is shared equally among the other outputs.|Float|
|outputFilePrefix|No|No|All output files created by the program will have this text as a prefix.|String|
|dropoutFraction|No|No|For training a single model; this is the value used for all dropout layers in the model|Float|
|convolutionLayersPerBlock|No|No|For training a single model; the number of convolutional layers in a block. All convolutional layers will have the same kernal size and number of filters in a block|Integer|
|extraFirstBlock|No|No|For training a single model; whether a extra block with no output will be put between the input layer and the block containing the first output (yes) or not (otherwise)|Boolean|
|initalLayerFilterCount|No|No|For training a single model; the filter count for the convolutional layers used in the first block|Integer|
|filterCountBlockMultiplicativeFactor|No|No|For training a single model; the filter count of convolutional layers in a block will be equal to the filter count of the convolutional laywers in the preceeding block multipled by this value (filter count will be rounded down to the nearest integer).|Float|
|initalLayerKernalSize|No|No|For training a single model; the convolutional layer kernal size used in the first block.|Integer|
|kernalSizeBlockMultiplicitiveFactor|No|No|For training a single model; the kernal size of convolutional layers in a block is equal to the kernal size of convolutional layers in the preceeding block multipled by this value (kernal sizes however will be rounded up the next odd number with a minimum size of 3).|Float|
|learningRate|No|No|For training a single model; the learning rate used by the optimiser|Float|
|hyperparameterOptimisationMaximumEvaluations|No|No|For hyperparameter optimisation; the maximum numebr of hyperparameter combinations to try.|Float|
|gpuQuantity|No|No|The number of GPUs to train on. If this value is less than 2 and no GPU exists it will be run on a CPU.|Integer|
|earlyStoppingMinDelta|No|No|In early stopping, an improvement of the monitored loss needs to be equal or greater than this value to reset the early stopping patience count.|Float|
|earlyStoppingPatience|No|No|The number of epochs that no improvement for the monitored loss can occur before early stopping for the current output occurs.|Integer|


### Hyperparameter limits configuration (hyperparameterLimitsConfiguration.txt)

|Name|Can have multiple values|Can occur multiple times|Description|Data type|
|----|------------------------|------------------------|-----------|---------|
|minimumDropoutFraction|No|No|The lower limit for dropoutFraction|Float|
|maximumDropoutFraction|No|No|The upper limit for dropoutFraction|Float|
|possibleConvolutionLayersPerBlock|Yes|No|Possible values for convolutionLayersPerBlock|Integer|
|possibleInitalLayerFilterCount|Yes|No|Possible values for initalLayerFilterCount|Integer|
|possibleFilterCountBlockMultiplicativeFactor|Yes|No|Possible values for filterCountBlockMultiplicativeFactor|Float|
|possibleInitalLayerKernalSize|Yes|No|Possible values for initalLayerKernalSize|Integer|
|possibleKernalSizeBlockMultiplicitiveFactor|Yes|No|Possible values for kernalSizeBlockMultiplicitiveFactor|Float|
|minimumLearningRate|No|No|The lower limit for learningRate|Float|
|maximumLearningRate|No|No|The upper limit for learningRate|Float|
