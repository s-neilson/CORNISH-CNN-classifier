# CORNISH-CNN-classifier
A convolutional neural network for the classification of astronomical obejcts using data that is part of the CORNISH (Co-Ordinated Radio 'N' Infrared Survey for High-mass star formation) project (website: http://cornish.leeds.ac.uk/public/index.php).

                                                                                                                                         
                                                                                                                                          
## Configuration files

All file paths are relative to the location of the file downloader. Configuration names and their values are seperated by an equal sign. For configuration parameters that take multiple values each values is seperated by an equals sign like the configuration name and first value. Some configuration names can occur multiple times in a configuration file. Boolean data types interpret as string "yes" as true, all other strings are interpreted as false.


### File downloader configuration (downloadConfiguration.txt)

|Name|Can have multiple values|Can occur multiple times|Description|Data type|
|----|------------------------|------------------------|-----------|---------|
|objectTypeFilePath|No|No|Path to CSV file contains pairs of object names and their associated type.|String|
|outputFileLocation|No|No|Path to the folder containing all of the downloaded files; this folder will have subfolders for every object type.|String|
|fileSourceURLPrefix|No|No|URL prefix for downloaded files; full download URL contains the prefix followed by the object name followed by a suffix depending on the specific file associated with each object.|String
|downloadPartialObjects|Whether if as many files for each object will be downloaded as possible (yes) or if an object will be skipped if one or more of it's files cannot be downloaded (otherwise).|Boolean|
|downloadFileSuffix|No|Yes|The suffixes of files that are to be downloaded.|String|

An example of a file URL to download with a fileSourceURLPrefix of http://cornish.leeds.ac.uk/public/data_src/, a name of G009.9702-00.5292 and a downloadFileSuffix of _CORNISH_5GHz.fits is http://cornish.leeds.ac.uk/public/data_src/G009.9702-00.5292_CORNISH_5GHz.fits


### Training input configuration (inputConfiguration.txt)

|Name|Can have multiple values|Can occur multiple times|Description|Data type|
|----|------------------------|------------------------|-----------|---------|
|trainSingleModel|No|No|Whether a single model should be trained (yes) or hyperparameters optimised (otherwise).|Boolean|
|desiredImageSize|No|No|Square edge size in pixels that input images should be scaled to.|Integer|
|desiredImageFov|No|No|Angular field of view that the input images should be cropped to.|Float|
|contigiousEqualAreaRejectionThreshold|No|No|Images that have a contigious area of the same pixel value connected to the image edge will be rejected if the areforementioned area equals or exceeds this value. This is done to remove images that may come from the edge of an astronomical survey; such images may cause problems with training.|Integer|
|objectLeafLabelTotalQuantity|No|No|The maximum amount of imaged obejcts that will exist for each object type. If the amount of loaded objects of a particular object type does not reach this limit, extra objects will be created from the loaded ones using the technique of data augmentation. No new objects will be created if the amount of loaded objects for a particular object type equals or exceeds this limit.|Integer|
|transformedObjectImageRemovalChance|No|No|Chance of an individual image of an augmented (or transformed) object to be replaced with a blank image. This is done so the classifier is forced not to rely on the presence of certain image channels.|Float|
|useFileSuffix|No|Yes|Specifies a file suffix (file name without the object name component) for images that can be used in training.|String|
|allowedObjectType|Yes|Yes|Ther first value specifies the object type based on object type folders that are created using the file downloader, while the remaining values specify in order the label hierarchy for this object type.|String|

