import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd
import random
import csv
import os
from ast import literal_eval

#HYPERPARAMS
K = 5;  # number of applicant clusters (tune these hyperparams)
L = 3;  # number of property characteristics (tune these hyperparams)
d1 = 10;  # dimension of applicant demographics
d2 = 5;  # dimension of property characteristics

############################ TRY 3 on 3/23 ###################
#       ********* randomly generate M1, which is a K x 10-dimensional (d1-D) points which will serve as the centers of the clusters
clusterCenterRange= 100
M1 = clusterCenterRange*np.random.uniform(-1, 1, size=(K,d1)) #clusterCenterRange affects the vals inside matrix
M2 = clusterCenterRange*np.random.uniform(-1, 1, size=(L,d2)) #clusterCenterRange affects the vals inside matrix
clusterStDevRange= 10
S1 = clusterStDevRange*np.random.uniform(0, 1, size=(K,d1)) #clusterCenterRange affects the vals inside matrix
#print(S1)
#           ********* randomly generate S1
S2 = clusterStDevRange*np.random.uniform(0, 1, size=(L,d2)) #clusterCenterRange affects the vals inside matrix
#N1, N2 = randomly generate number of elements per cluster (two vectors that are length K and length L)
#******because appending to numpy arrays is slow, will make list first then use it to construct an np.array
N1 = []
for num in range(K):
    x = random.randint(2,10)
    N1.extend([x])
# N1=np.array(N1)
N1 = np.array(N1)
#print(N1)
N2 = []
for num in range(L):
    x = random.randint(2,10)
    N2.extend([x])
N2 = np.array(N2)
#print(N2)
# **************** V will be made from numpy.random.normal.
#numpy.random.normal uses np.random.normal(M1, S1, N1) where:
#
# M1 is array_like of floats
# Mean (“centre”) of the distribution.
#
# S1 is array_like of floats
# Standard deviation (spread or “width”) of the distribution. Must be non-negative.
#
# N1 is output shape which specifies the number of elements per cluster
#
# Returns newly made clusters which are:
# Drawn samples from the parameterized normal distribution.

V = {}

for clusterNum in range(K): #have to populate each of the K clusters
    numIndividuals = N1[clusterNum] #get specified num of individuals in the current cluster
    indivCluster = np.ndarray(shape=(numIndividuals, d1),dtype='f') #make a cluster list
    for dem in range(d1): #for each demographic
        indivCluster[:,dem] = np.random.normal(M1[clusterNum][dem], S1[clusterNum][dem], numIndividuals)
        # print(indivCluster[:,dem])

    indivCluster.astype(float)
    V[clusterNum]=indivCluster
    print(indivCluster)
    print("type of indivCluster is "+str(type(indivCluster)))
    # print("type of V.get(4) is "+str(type(V.get(4))))
    # print("V.get(4) is "+V.get(4))
U = {}

for clusterNum in range(L): #have to populate each of the L clusters
    numProperties = N2[clusterNum] #get specified num of individuals in the current cluster
    propertyCluster = np.ndarray(shape=(numProperties, d2)) #make a cluster matrix
    for ch in range(d2): #for each characteristic
        propertyCluster[:,ch] = np.random.normal(M2[clusterNum][ch], S2[clusterNum][ch], numProperties)
        #print("propertyCluster is ")
        #print(propertyCluster)
        U[clusterNum]=propertyCluster

for clusterNum in range(L):
    #print("**********************************************************")
    #print("**********************************************************")
    #print("current cluster num is "+ str(clusterNum))
    #print("**********************************************************")
    #print("**********************************************************")

    c = U.get(clusterNum)
    j =0
    for property in c:
        #print(str(j)+"th property has the following values for the 10 characteristics: " )
        i =0
        j=j+1
        for characteristic in property:
            #print("value of characteristic "+str(i)+" is:")
            #print(characteristic)
            i=i+1


#we have 2 matrices, V and U. we want to randomly match each entry in U to an entry in V.
#so we want to iterate through each individual from each cluster from V.
#for each of these individuals, we randomly select a property from U, the np array containing the clusters of properties
numIndividuals = np.sum(N1)
applications = []
lenderArray = np.array([0,1,2,3])
approvalArray = np.array([0,1])

allClusters = V.keys()
#print("print")
#because appending to numpy arrays is slow, will make list first then use it to construct an np.array
counter = 0
for clusterName in allClusters:

    cluster = V.get(clusterName)
    for individual in cluster:
        keyList = list(U.keys())
        # print(U)
        randomPropClusterInd = random.randint(0,len(keyList)-1)
        randomPropCluster = U.get(keyList[randomPropClusterInd]) #get a cluster of properties
        #print("randomPropCluster is ")
        #print(randomPropCluster)
        randomPropInd = random.randint(0,len(randomPropCluster)-1)
        randomProperty = randomPropCluster[randomPropInd] #replace =True means that the same property can be selected multiple times
        #print(randomProperty)
        counter=counter+1
#np.random.choice generates samples from uniform random sample

        randomLender = np.random.choice(lenderArray, replace=True)
        randomApproval = np.random.choice(approvalArray, replace=True) #np.random.choice generates samples from uniform random sample
        print("type of indivCluster is "+str(type(individual)))
        arrIndividual = np.empty(shape = d1, dtype='f')
        for ele in individual:
            np.append(arrIndividual,ele)

        newTup = (counter, arrIndividual, randomProperty, randomLender, randomApproval)
        applications.append((newTup))#add as nested element to the array



applications=np.array(applications, dtype='O')
# for app in applications:
    #print("an app is ")
    #print(app)

df = pd.DataFrame(list())
df.to_csv('empty_csv.csv')

f = open('results/r.csv', 'w') #write
#df['col2'] = df['col2'].apply(literal_eval) try to use this to fix dataframe problem
# create csv writer
write = csv.writer(f)
title = ['Num Applicant','Applicant Information', 'Property', 'Lender', 'Approval Status']
write.writerow(title)
write.writerows(applications)

#os.rename('results/empty_csv.csv', 'results/filled_csv.csv')
