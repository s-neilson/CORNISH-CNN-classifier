from astropy.nddata import Cutout2D
from scipy.misc import imresize

#Removes all but the first two axes (the axes that correspond to the image dimensions) from a peice of WCS data if they exist.
def wcsRemoveNonImageAxes(wcsData):
    currentWcsData=wcsData

    while(currentWcsData.naxis>2):
        currentWcsData=currentWcsData.dropaxis(-1)# Removes the last axis in the axis list.
    
    return currentWcsData

#Gets the field of view from the WCS data associated with a FITS file
def wcsGetFov(wcsData):
    footprint=wcsData.calc_footprint() #This is the coordinates of the four corners of the image.
    return abs(footprint[0][1]-footprint[1][1])


# Crops and resizes the data of a fits file to a particular field of view and image dimension.
def imageMatchFovAndSize(imageData,wcsData,finalSize,finalFov):
    finalShape=(finalSize,finalSize)
    fov=wcsGetFov(wcsData)
    resizeFactor=finalFov/fov

    centrePoint=(int(0.5*(imageData.shape[0])),int(0.5*(imageData.shape[1])))#The centre point of the original image.
    newSize=int(imageData.shape[0]*resizeFactor) #The size of the cutout section in pixels
    
    cutout=Cutout2D(data=imageData,wcs=wcsData,position=centrePoint,size=newSize) #The area is cut out of the image
    cutoutData=cutout.data

    resizedData=imresize(cutoutData,finalShape,mode="F")
    return resizedData