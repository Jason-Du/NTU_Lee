import pandas as pd
import os
import numpy as np




def getShuffleData(arrayX):
    arrayRandomIndex = np.arange(len(arrayX))
    np.random.shuffle(arrayRandomIndex)
    return arrayX[arrayRandomIndex]



test=[[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
test=np.array(test)
print(test)
testx=test



testx=getShuffleData(testx)
print(testx)



arrayRandomIndex = np.arange(len(test))
print(type(arrayRandomIndex))

testy=test
index=np.array([2,3, 1, 0])
print(type(index))
print(test[index])

print("---------------")
print(np.zeros(1))