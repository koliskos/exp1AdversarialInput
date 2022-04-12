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
#arr_of_full_apps=[['App_Number'], ['App_Feature_0'],['App_Feature_1'],['App_Feature_2'],['App_Feature_3'],['App_Feature_4'],['App_Feature_5'],['App_Feature_6'],['App_Feature_7'],['App_Feature_8'],['App_Feature_9'],['Prop_Feature_0'],['Prop_Feature_1'],['Prop_Feature_2'],['Prop_Feature_3'],['Prop_Feature_4'], ['Lender'], ['Approval_Status']]




for i in range(num_rows):
    numApp=df.iat[i,0]
    appInfo=df.iat[i,1]
    propInfo=df.iat[i,2]
    lender=df.iat[i,3]
    approval =df.iat[i,4]

    appLen = len(appInfo)

    # nowFloatsArray=np.empty(10)
    appFloatsArray=[]
    propertyAsArr=[]
    appInfo= appInfo[1: appLen-1]
    appInfo_as_float = appInfo.split(',')

    counter = 0

    for i in range(len(appInfo_as_float)):
        ele=appInfo_as_float[i]
        print("ele is "+str(ele))
        # ele.replace(" ", "")
        # print (type(ele))
        if ele[0] == '-':
            eleNeg = ele[1:]
            # print(ele)
            # print(eleNeg)
            appNum = float(eleNeg)*-1
            # print(type(now))
            appFloatsArray.append(appNum)
        else:
            appNum = float(ele)
            appFloatsArray.append(appNum)
        counter+=1
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
    another_full_app = [numApp, appFloatsArray[0], appFloatsArray[1],appFloatsArray[2],appFloatsArray[3],appFloatsArray[4],appFloatsArray[5],appFloatsArray[6],appFloatsArray[7],appFloatsArray[8],appFloatsArray[9],propertyAsArr[0], propertyAsArr[1],propertyAsArr[2],propertyAsArr[3],propertyAsArr[4],lender, approval]
    arr_of_full_apps.append(another_full_app)
print(arr_of_full_apps)

df_clean = pd.DataFrame (arr_of_full_apps, columns = ['App_Number', 'App_Feature_0','App_Feature_1','App_Feature_2','App_Feature_3','App_Feature_4','App_Feature_5','App_Feature_6','App_Feature_7','App_Feature_8','App_Feature_9','Prop_Feature_0','Prop_Feature_1','Prop_Feature_2','Prop_Feature_3','Prop_Feature_4', 'Lender', 'Approval_Status']
)
print (df_clean)

    #     row_1_as_int.append(-1*float(eleNeg))
    # else:
    #     row_1_as_int.append(float(ele))


# print(row_1_as_int)
