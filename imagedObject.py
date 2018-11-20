import copy
import os
from astropy.io import fits
from astropy import wcs
from astropy.visualization import MinMaxInterval
import numpy
import keras_preprocessing

import fitsFileFunctions




class ImagedObject:
    name=""
    label=()
    
    imageSize=0
    imageData=None #All of the data from the fits files in a 3d numpy array.   
    nonBlankImageCount=0
    
        
#Class for creating an ImagedObject from a set of file paths.   
class FileImagedObject(ImagedObject):        
    def __init__(self,filePaths,name,label,imageSize,imageFov,rejectionThresholdArea):
        self.name=name
        self.label=label
        self.imageSize=imageSize
        
        
        #The imageData array is initally set to zeros so that images that are either missing, not square in shape, or have been determined to be from the survey edge will have a blank entry in imageData.
        # A 32 bit data type is sufficient as the input .fits files are 32 bit images themselves.
        self.imageData=numpy.zeros(shape=(self.imageSize,self.imageSize,len(filePaths)),dtype=numpy.float32)
               
        
        for currentIndex,currentFilePath in enumerate(filePaths):
            self.__loadFitsFile(currentIndex,currentFilePath,imageSize,imageFov,rejectionThresholdArea)  
            
            
    def __loadFitsFile(self,imageDataIndex,filePath,imageSize,imageFov,rejectionThresholdArea):
        if(filePath!=""): #If the file corresponding to currentIndex was found in the folder.                 
            newHDU=fits.open(filePath)[0]
            newImageData=newHDU.data
            newWCS=wcs.WCS(newHDU.header)
            newWCS=fitsFileFunctions.wcsRemoveNonImageAxes(newWCS)

            if(newImageData.shape[0]!=newImageData.shape[1]):
                return #The image is not square in shape and is therefore rejected.
                
            if(not self.__imageFromEdgeOfSurvey(newImageData,rejectionThresholdArea)):
                newImageData=fitsFileFunctions.imageMatchFovAndSize(newImageData,newWCS,imageSize,imageFov)
                self.nonBlankImageCount+=1
                
                normalizerInterval=MinMaxInterval()
                normalizedNewImageData=normalizerInterval(newImageData)#The pixel values in the curent image are normalized.

                self.imageData[:,:,imageDataIndex]=normalizedNewImageData #The image has been successfully loaded.
     
    #Uses a modified version of the flood fill algorithm (used for changing the colour of areas in art software) in order to detect large regions that have the same pixel value that originate
    # from the edge; regions that may indicate that the image comes from the edge of an astronomical survey, meaning it should be rejected.
    def __imageFromEdgeOfSurvey(self,image,rejectionThresholdArea):
        edgePixels=() #A tuple that holds the coordinates for parts of the image at the edge. Areas from the edge of a survey will have large sections all with the same pixel value, and they always begin at the edge.

        for i in range(0,image.shape[0]):
            edgePixels+=((0,i),) #Left edge
            edgePixels+=((image.shape[0]-1,i),) #Right edge
            edgePixels+=((i,0),) #Bottom edge
            edgePixels+=((i,image.shape[0]-1),) #Top edge
            
        for currentEdgeLocation in edgePixels: #All edge locations are checked for being part of a large region of the same value.
            visitedLocations=numpy.full(shape=image.shape,fill_value=False) #Stores locations that have already been visited.
            currentPixelValue=image[currentEdgeLocation]

            if(self.__exploreNeighbourPixels(image,currentPixelValue,0,rejectionThresholdArea,currentEdgeLocation,visitedLocations)):
                return True #A large area of contigious colour originates from the pixel at currentEdgeLocation.
            
        return False #The image has not been determined to be from the edge of a survey.
    
    #Recursively explores neighbouring pixels keeping a count of the current area of contigious value until either a threshold area is reached or the contigious area is found to be smaller than the threshold area.
    def __exploreNeighbourPixels(self,image,targetPixelValue,currentArea,thresholdArea,currentLocation,visitedLocations):
        outOfXBounds=(currentLocation[0]<0) or (currentLocation[0]>=image.shape[0])
        outOfYBounds=(currentLocation[1]<0) or (currentLocation[1]>=image.shape[0])
        
        if(outOfXBounds or outOfYBounds):
            return False #The pixel at currentLocation does not exist.
        
        if(visitedLocations[currentLocation]==True):
            return False #The pixel at currentLocation has already been accounted for.
        else:
            visitedLocations[currentLocation]=True       
        
        if(image[currentLocation]==targetPixelValue):
            currentArea+=1
        else:
            return False #This pixel is not part of the contigious area.
   
        if(currentArea>=thresholdArea):
            return True #The area of contigious value has exceeded the threshold; this result will be pass to calling functions recursively.
            
        visitedLocations[currentLocation]=True
        
        upNeighbourLocation=(currentLocation[0],currentLocation[1]+1)
        downNeighbourLocation=(currentLocation[0],currentLocation[1]-1)
        leftNeighbourLocation=(currentLocation[0]-1,currentLocation[1])
        rightNeighbourLocation=(currentLocation[0]+1,currentLocation[1])
        neighbourLocations=(upNeighbourLocation,downNeighbourLocation,leftNeighbourLocation,rightNeighbourLocation)
        
        for i in neighbourLocations:
            if(self.__exploreNeighbourPixels(image,targetPixelValue,currentArea,thresholdArea,i,visitedLocations)):
                return True #The threshold area has been reached by a pixel contigious to this one.
        
        return False #The threshold area has not been reached by a pixel contigious to this one.


#Class for creating an ImagedObject from .fits files in a folder.
class FolderImagedObject(FileImagedObject):    
    def __init__(self,folderPath,name,label,imageSize,imageFov,rejectionThresholdArea,fileSuffixes):
        self.name=name
        self.label=label
        self.imageSize=imageSize
        
        currentFolderContents=os.listdir(folderPath)
        filesToLoad=self.__getFilesToLoad(folderPath,currentFolderContents,fileSuffixes)
        super().__init__(filesToLoad,name,label,imageSize,imageFov,rejectionThresholdArea) #Uses FileImagedObject to create the object from the file list.        


    #Returns a list of the filenames of .fits files to be loaded for each corresponding image that is to be in the imageData array.
    def __getFilesToLoad(self,folderPath,currentFolderContents,fileSuffixes):        
        filesToLoad=[""]*len(fileSuffixes) #.fits files that are missing will cause their corresponding entry to remain blank.
        
        for currentContentName in currentFolderContents:
            currentContentPath=folderPath+"/"+currentContentName
            
            if((os.path.isfile(currentContentPath)) and (os.path.splitext(currentContentPath)[1]==".fits")): #Locates all .fits files in the folder.
                currentFileName=os.path.split(currentContentPath)[1]
                
                possibleImageNames=[self.name+currentSuffix for currentSuffix in fileSuffixes]
                currentImageIndex=0
                
                try:
                    currentImageIndex=possibleImageNames.index(currentFileName) #Gets the location in the imageData array that the data from the current .fits file should occupy
                    filesToLoad[currentImageIndex]=currentContentPath 
                except ValueError:
                    pass #The ValueError means that the data in the current .fits file should not be included in the imageData array. However, the program can still continue.
        
        return filesToLoad




class TransformedImagedObject(ImagedObject):
    def __init__(self,originalObject,imageRemovalChance):
        self.name=originalObject.name
        self.label=originalObject.label
        self.imageData=copy.deepcopy(originalObject.imageData)
        
        for i in range(0,self.imageData.shape[2]): #Each image has a chance of being removed.
            if(numpy.random.random_sample(size=None)<imageRemovalChance):
                self.imageData[:,:,i]=numpy.zeros_like(self.imageData[:,:,0]) #An image is removed.
                self.nonBlankImageCount-=1

        #In order to make transformed images more natural, without sharp borders to areas where pixels have been filled with 
        #zero values, the average value of each image is put in the places where the filled zero value pixels would occur.
        #This is done by subtracting the average background value from each image, and then adding it again when the images
        #are transformed. This causes the transformed content of the image to be returned to their original values and the
        #areas filled with zero value pixels to have the average background value.
        
        imageBackgroundValues=numpy.average(self.imageData,axis=(0,1))
        imageBackgrounds=numpy.empty_like(self.imageData)
        
        for i in range(0,self.imageData.shape[2]):
            imageBackgrounds[:,:,i]=numpy.full_like(self.imageData[:,:,0],fill_value=imageBackgroundValues[i])
            
        self.imageData=numpy.subtract(self.imageData,imageBackgrounds) #The backgrounds are subtracted from each image.
        
    
        if(numpy.random.random_sample(size=None)>0.5): #Vertical flip
            self.imageData=keras_preprocessing.image.flip_axis(self.imageData,0)
            
        if(numpy.random.random_sample(size=None)>0.5): #Horizontal flip
            self.imageData=keras_preprocessing.image.flip_axis(self.imageData,1)
            
        
        self.imageData=keras_preprocessing.image.random_shift(self.imageData,0.2,0.2,row_axis=0,col_axis=1,channel_axis=2,fill_mode="constant",cval=0.0)
        self.imageData=keras_preprocessing.image.random_rotation(self.imageData,360.0,row_axis=0,col_axis=1,channel_axis=2,fill_mode="constant",cval=0.0)
        self.imageData=keras_preprocessing.image.random_zoom(self.imageData,(0.8,1.2),row_axis=0,col_axis=1,channel_axis=2,fill_mode="constant",cval=0.0)
        
        self.imageData=numpy.add(self.imageData,imageBackgrounds) #The backgrounds are added to the images again.
        
               