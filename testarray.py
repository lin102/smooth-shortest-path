
import numpy as np

import os
import math

y = np.full((5,6), np.nan)
shape = y.shape



visitUnvisitArr = np.zeros((2,3), dtype = np.int)
allvisitedArr = np.full((2, 3), 10,dtype = np.int)
visitUnvisitArr[0][0] = 10
visitUnvisitArr[0][1] = 10
visitUnvisitArr[0][2] = 10
visitUnvisitArr[1][0] = 10
visitUnvisitArr[1][1] = 555
visitUnvisitArr[1][2] = 666
visitUnvisitArr==allvisitedArr
print(visitUnvisitArr)
print(allvisitedArr)
print((visitUnvisitArr==allvisitedArr).any())

preVertex = [[[] for col in range(5)] for row in range(6)]
j = 1
i = 2
preVertex[1][2].append(j)
preVertex[1][2].append(i)
print (preVertex)
preVertex[1][2]
arr = [['1' for col in range(5)] for row in range(3)]

preVertex = [[[] for col in range(3)] for row in range(2)]
print(preVertex)
print(preVertex[1][1] != [])

demonstrate_array = np.full((5,6),np.nan)
print(demonstrate_array)


p = 1
while(p<5):
    p += 1
    if p == 2:
        continue
else:
    print('p',p)
    print('I am out')



