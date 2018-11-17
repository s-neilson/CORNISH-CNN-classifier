import csv



class Configuration:
    __configurationDictionary=None
    __filePath=""
    __delimiterCharacter=""
    
    def __init__(self,filePath,delimiterCharacter):
        self.__filePath=filePath
        self.__delimiterCharacter=delimiterCharacter
        self.__loadConfiguration()
    
    #Gets a loaded configuration value from the configuration dictionary and casts it if necessary. 
    def getConfigurationValue(self,configurationKey,outputType):
        rawValue=self.__configurationDictionary[configurationKey]
        
        #Casts either every element of rawValue individually if it is a tuple of values or just the rawValue otherwise.
        rawValueCaster=lambda castTypeFunction:[castTypeFunction(i) for i in rawValue] if(type(rawValue)==tuple) else castTypeFunction(rawValue)
        
        boolCaster=lambda inputValue:inputValue=="yes"
        
        if(outputType=="raw"):
            return rawValue #No casting is done on the output
        elif(outputType=="bool"):
            return rawValueCaster(boolCaster)
        elif(outputType=="int"):
            return rawValueCaster(int)
        elif(outputType=="float"):
            return rawValueCaster(float)
        
        raise Exception("Invalid desired outputType "+outputType+" for getConfigurationValue")
    
    #Gets pairs of all loaded configuration keys and their values.
    def getAllConfigurations(self):
        return self.__configurationDictionary.items()
          
    #Loads all the entries from a configuration file
    def __loadConfiguration(self):
        configurationFile=open(self.__filePath,"r")
        configurationReader=csv.reader(configurationFile,delimiter=self.__delimiterCharacter)
        self.__configurationDictionary={}

        for currentLine in configurationReader:
            if(len(currentLine)==0):
                continue #Allows the spacing of sections of the configuration file

            currentKey=currentLine[0]      
            currentValue=None
        
            if(len(currentLine)==2):
                currentValue=currentLine[1] #For a single value to a key.
            else:
                currentValue=tuple(currentLine[1:]) #For multiple values for a key.        

            if(currentKey in self.__configurationDictionary):
                if(type(self.__configurationDictionary[currentKey])!=list):
                    self.__configurationDictionary[currentKey]=[self.__configurationDictionary[currentKey]] #Transforms the value into list form.
                
                self.__configurationDictionary[currentKey].append(currentValue) #For non unique keys.
            else:
                self.__configurationDictionary[currentKey]=currentValue #For a unique key.
     
    
        configurationFile.close()        
    
    
        