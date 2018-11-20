from collections import OrderedDict


#Returns a list containing the possible object labels at every level in the object label hierarchy
#given an list of the labels associated with each object. Also returns how deep the object label
#hierarchy tree is.
def getObjectHierarchyLabels(labelList):
    #Each object type needs to have the same depth in the object heirarchy tree, this is checked below.
    hierarchyDepth=len(labelList[0])
    for currentLabel in labelList:
        if(len(currentLabel)!=hierarchyDepth):
            raise Exception("Depth of object type label hierarchy is not the same for all object types.")
    
    
    #Below creates sets of unique labels at each depth level in the object type hierarchy.
    labelSets=[]
    for i in range(0,hierarchyDepth):
        currentDepthLabelList=[currentLabel[i] for currentLabel in labelList] #List of the labels for all objects at a particular depth in the hierarchy.
        labelSet=OrderedDict().fromkeys(currentDepthLabelList) #The dictionary is used to remove the duplicates; it is an ordered dictionary to ensure the labels are always in the same 
        #order whenever the function is run on the same input (using Python's inbuilt set() does not guarantee this).
        labelSets.append(list(labelSet.keys()))
            
    return labelSets,hierarchyDepth