import os
import requests
from joblib import Parallel
from joblib import delayed

from loadConfiguration import Configuration


def containsOkStatusCode(response):
    return response.status_code==200 #HTTP status code 200 means "OK".

#Function that returns a list of the URLs for files to download for a particular object given it's name and target folder path.
def getFilesToDownloadForObject(objectName,objectFolderPath,fileSuffixes,urlPrefix):
    filesToDownload=[]
    
    for i in fileSuffixes:
        currentSuffix=i.strip() #Whitespace is removed from end of line
        filePath=objectFolderPath+"/"+objectName+currentSuffix

        if(os.path.isfile(filePath)==False): #If the file is missing
            currentURL=""+urlPrefix+objectName+currentSuffix
            filesToDownload.append([currentURL,objectName+currentSuffix])
            
    return filesToDownload


#Returns if a certain URL returns a status code of 200, meaning it can be downloaded.
def okHeadResponse(url,requestSession):
    headResponse=requestSession.head(url)
    
    return containsOkStatusCode(headResponse)

def downloadFile(url,requestSession):
    response=requestSession.get(url)
    return response

#Returns a list of downloaded files and names that the files should be saved as.
def downloadFiles(fileList,requestSession,partialDownloadAllowed):
    downloadedFileResponses=[]
    numberOfFiles=len(fileList)
    fileURLs=[i[0] for i in fileList]
    fileNames=[i[1] for i in fileList]
    
    #Done to see initally if all of the files exist.    
    if(not partialDownloadAllowed):
        headResponses=Parallel(n_jobs=numberOfFiles,prefer="threads")(delayed(okHeadResponse)(i,requestSession) for i in fileURLs)
        
        if(False in headResponses):  #If not all the files can be downloaded
            return None
        
    downloadedFileResponses=Parallel(n_jobs=numberOfFiles,prefer="threads")(delayed(downloadFile)(i,requestSession) for i in fileURLs)
    return zip(downloadedFileResponses,fileNames)   




def main():
    configuration=Configuration(os.getcwd()+"/configurations/downloadConfiguration.txt","=")

    objectTypeFilePath=os.getcwd()+configuration.getConfigurationValue("objectTypeFilePath","raw")
    outputFileLocation=os.getcwd()+configuration.getConfigurationValue("outputFileLocation","raw")
    fileSourceURLPrefix=configuration.getConfigurationValue("fileSourceURLPrefix","raw")
    objectFileSuffixes=configuration.getConfigurationValue("downloadFileSuffix","raw")
    downloadPartialObjects=configuration.getConfigurationValue("downloadPartialObjects","bool")

    
    objectTypes=Configuration(objectTypeFilePath,",")        
    objects=[(currentName,currentObjectType) for currentName,currentObjectType in objectTypes.getAllConfigurations()] #Contains pairs of object names and their corresponding object type.
     
    
    print("There are "+str(len(objects))+" objects in total, each with "+str(len(objectFileSuffixes))+" associated files.")
    print("Partial download of objects "+("enabled" if(downloadPartialObjects) else "disabled"))



    session=requests.Session()

    for currentIndex,currentObject in enumerate(objects): #Loops through all objects
        currentObjectName=currentObject[0]
        currentObjectFolderPath=outputFileLocation+"/"+currentObject[1]+"/"+currentObject[0]

        print("Object "+str(currentIndex+1)+"/"+str(len(objects))+", Detecting missing files for "+currentObjectName+", ",)
        filesToDownload=getFilesToDownloadForObject(currentObjectName,currentObjectFolderPath,objectFileSuffixes,fileSourceURLPrefix)
    
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
                if(containsOkStatusCode(currentDownloadedFile[0])): #If the file can be downloaded.
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


main()
