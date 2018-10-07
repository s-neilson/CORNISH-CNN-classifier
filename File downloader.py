import os
import csv
import requests
from joblib import Parallel
from joblib import delayed

from loadConfiguration import Configuration





configuration=Configuration(os.getcwd()+"/downloadConfiguration.txt","=")

objectTypeFilePath=os.getcwd()+configuration.getConfigurationValue("objectTypeFilePath","raw")
outputFileLocation=os.getcwd()+configuration.getConfigurationValue("outputFileLocation","raw")
fileSourceURLPrefix=configuration.getConfigurationValue("fileSourceURLPrefix","raw")
objectFileSuffixes=configuration.getConfigurationValue("downloadFileSuffix","raw")
downloadPartialObjects=configuration.getConfigurationValue("downloadPartialObjects","bool")

    
objectTypes=Configuration(objectTypeFilePath,",")        
objects=[] #Contains pairs of object names and the corresponding paths that their files will be located at.

for currentName,currentObjectType in objectTypes.getAllConfigurations(): 
    currentObjectFolderPath=outputFileLocation+"/"+currentObjectType+"/"+currentName
    objects.append((currentName,currentObjectFolderPath))
     
    
print("There are "+str(len(objects))+" objects in total, each with "+str(len(objectFileSuffixes))+" associated files.")
print("Partial download of objects "+("enabled" if(downloadPartialObjects) else "disabled"))





def downloadOkStatusCode(response):
    return response.status_code==200 #HTTP status code 200 means "OK".

#Function that returns a list of the URLs for files to download for a particular object.
def getFilesToDownloadForObject(currentObject):
    objectName=currentObject[0]
    objectFolderPath=currentObject[1]
    filesToDownload=[]
    
    for i in objectFileSuffixes:
        currentSuffix=i.strip() #Whitespace is removed from end of line
        filePath=objectFolderPath+"/"+objectName+currentSuffix

        if(os.path.isfile(filePath)==False): #If the file is missing
            currentURL=""+fileSourceURLPrefix+currentObjectName+currentSuffix
            filesToDownload.append([currentURL,currentObjectName+currentSuffix])
            
    return filesToDownload

def headResponse(url,requestSession):
    headResponse=requestSession.head(url)
    
    return downloadOkStatusCode(headResponse)

def downloadFile(url,label,requestSession):
    response=requestSession.get(url)
    return (response,label)

def downloadFiles(fileList,requestSession,partialDownloadAllowed):
    downloadedFileResponses=[]
    numberOfFiles=len(fileList)
    
    argList=[]
    for i in fileList:
        argList.append((i[0],i[1],requestSession))
        
    #Done to see initally if all of the files exist.    
    if(not partialDownloadAllowed):
        headResponses=Parallel(n_jobs=numberOfFiles,prefer="threads")(delayed(headResponse)(i[0],requestSession) for i in argList)
    
        for i in headResponses:
            if(i==False): #If not all the files can be downloaded
                return None

    downloadedFileResponses=Parallel(n_jobs=numberOfFiles,prefer="threads")(delayed(downloadFile)(*args) for args in argList)

    return downloadedFileResponses   





session=requests.Session()
currentObjectCount=0

for currentObject in objects: #Loops through all objects
    currentObjectCount+=1
    currentObjectName=currentObject[0]
    currentObjectFolderPath=currentObject[1]

    print("Object "+str(currentObjectCount)+"/"+str(len(objects))+", Detecting missing files for "+currentObjectName+", ",)
    filesToDownload=getFilesToDownloadForObject(currentObject)
    
    numberOfFilesToDownload=len(filesToDownload)
    if(numberOfFilesToDownload==0):
        print("All files present, no files to download for this object\n")
        continue
    else:
        print(str(numberOfFilesToDownload)+" will be downloaded for this object, ",)
    
    downloadedFiles=downloadFiles(filesToDownload,session,downloadPartialObjects)
    
    downloadedFileCount=0
    if(downloadedFiles==None):
        print("Partial download disabled, download of a file failed, object "+currentObjectName+" will be skipped\n")
    else:
        for currentDownloadedFile in downloadedFiles:
            if(downloadOkStatusCode(currentDownloadedFile[0])): #If the file can be downloaded
                currentFileData=currentDownloadedFile[0].content
   
                if(os.path.exists(currentObjectFolderPath)==False):
                    os.makedirs(currentObjectFolderPath) #Recursively creates the object folder if it doesn't already exist.
            
                currentOutputFilePath=currentObjectFolderPath+"/"+currentDownloadedFile[1]
                currentOutputFile=open(currentOutputFilePath,"wb")
                currentOutputFile.write(currentFileData)
                currentOutputFile.close()
                downloadedFileCount+=1
        
        print("Download of "+str(downloadedFileCount)+" files for this object successful\n")
    
print("Processing of objects is complete")

