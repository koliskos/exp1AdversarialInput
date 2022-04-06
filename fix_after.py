import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd
import random
import csv
import os
from ast import literal_eval



fil = 'https://raw.githubusercontent.com/koliskos/exp1AdversarialInput/main/r5.csv'
df = pd.read_csv(fil)
num_rows = df.shape[0]
#we know there are 5 columns
# arr = np.empty((num_rows,5), dtype=float)
arr_of_full_apps=[]

for i in range(num_rows):
    numApp=df.iat[i,0]
    appInfo=df.iat[i,1]
    propInfo=df.iat[i,2]
    lender=df.iat[i,3]
    approval =df.iat[i,4]







    appLen = len(appInfo)

    # nowFloatsArray=np.empty(10)
    nowFloatsArray=[]
    propertyAsArr=[]
    appInfo= appInfo[1: appLen-1]
    appInfo_as_float = appInfo.split(',')


    for ele in appInfo_as_float:
        # ele.replace(" ", "")
        # print (type(ele))
        if ele[0] == '-':
            eleNeg = ele[1:]
            # print(ele)
            # print(eleNeg)
            now = float(eleNeg)*-1
            # print(type(now))
            nowFloatsArray.append(now)
        else:
            now = float(ele)
            nowFloatsArray.append(now)
    # print(nowFloatsArray)


    propLen =len(propInfo)
    propInfo= propInfo[1: propLen-1]
    propInfo_as_float = propInfo.split()
    for ele in propInfo_as_float:
        # ele.replace(" ", "")
        # print (type(ele))
        if ele[0] == '-':
            eleNeg = ele[1:]
            # print(ele)
            # print(eleNeg)
            now = float(eleNeg)*-1
            # print(type(now))
            propertyAsArr.append(now)
        else:
            now = float(ele)
            propertyAsArr.append(now)


    # another = np.array([numApp, nowFloatsArray, property, lender, approval])
    another_full_app = [numApp, nowFloatsArray, propertyAsArr, lender, approval]
    arr_of_full_apps.append(another_full_app)
print(arr_of_full_apps[1])















    #     row_1_as_int.append(-1*float(eleNeg))
    # else:
    #     row_1_as_int.append(float(ele))


# print(row_1_as_int)
