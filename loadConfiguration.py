import csv

#Loads all the entries from a configuration file
def loadConfiguration(filePath,delimiterCharacter):
    configurationFile=open(filePath,"r")
    configurationReader=csv.reader(configurationFile,delimiter=delimiterCharacter)
    configurationDictionary={}

    for currentLine in configurationReader:
        if(len(currentLine)==0):
            continue #Allows the spacing of sections of the configuration file

        currentKey=currentLine[0]      
        currentValue=None
        
        if(len(currentLine)==2):
            currentValue=currentLine[1] #For a single value to a key.
        else:
            currentValue=tuple(currentLine[1:]) #For multiple values for a key.        
       
        if(currentKey in configurationDictionary):
            if(type(configurationDictionary[currentKey])!=list):
                configurationDictionary[currentKey]=[configurationDictionary[currentKey]] #Transforms the value into list form.
                
            configurationDictionary[currentKey].append(currentValue) #For non unique keys.
        else:
            configurationDictionary[currentKey]=currentValue #For a unique key.
     
    
    configurationFile.close()        
    return configurationDictionary