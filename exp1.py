import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import random

#HYPERPARAMS
K = 5;  # number of applicant clusters (tune these hyperparams)
L = 3;  # number of property characteristics (tune these hyperparams)
d1 = 10;  # dimension of applicant demographics
d2 = 5;  # dimension of property characteristics



############################ TRY 3 on 3/23 ###################
#       ********* randomly generate M1, which is a K x 10-dimensional (d1-D) points which will serve as the centers of the clusters
clusterCenterRange= 100
M1 = clusterCenterRange*np.random.uniform(-1, 1, size=(K,d1)) #clusterCenterRange affects the vals inside matrix
#make array of size k and d1

# for num in range(K): # however many points you want
#     newCenter = []
#     for n in range(d1): #extend vector by another dimension for each dim in d1. d1 is however many dimensions each point will have
#         x = random.randint(0, 2500)
#         newCenter.extend([x])
#     M1.append(newCenter) #new d1-dimensional point added as an individual vector into the matrix M1
    #M1 shall end up having K d1-dimensional points

#print(M1)

#           ********* randomly generate M2, which is an L x 5-dimensional (d2-D) points which will serve as the centers of the clusters


M2 = clusterCenterRange*np.random.uniform(-1, 1, size=(L,d2)) #clusterCenterRange affects the vals inside matrix
#make array of size L and d2
# for num in range(L): # however many points you want
#     newCenter = []
#     for n in range(d2): #extend vector by another dimension for each dim in d1. d1 is however many dimensions each point will have
#         x = random.randint(0, 2500)
#         newCenter.extend([x])
#     M2.append(newCenter) #new d1-dimensional point added as an individual vector into the matrix M1
#     #M1 shall end up having K d1-dimensional points

#print(M2)

#           ********* randomly generate S1, which has values of the std of each cluster???

clusterStDevRange= 10
S1 = clusterStDevRange*np.random.uniform(0, 1, size=(K,d1)) #clusterCenterRange affects the vals inside matrix

# for num in range(K): # however many points you want
#     newStandDev = []
#     for n in range(d1): #extend vector by another dimension for each dim in d1. d1 is however many dimensions each point will have
#         x = random.randint(0, 101)
#         newStandDev.extend([x])
#     S1.append(newStandDev) #new d1-dimensional point added as an individual vector into the matrix M1
#     #M1 shall end up having K d1-dimensional points

print(S1)
#           ********* randomly generate S1
S2 = clusterStDevRange*np.random.uniform(0, 1, size=(L,d2)) #clusterCenterRange affects the vals inside matrix





#N1, N2 = randomly generate number of elements per cluster (two vectors that are length K and length L)

#******because appending to numpy arrays is slow, will make list first then use it to construct an np.array

N1 = []
for num in range(K):
    x = random.randint(2,10)
    N1.extend([x])
N1=np.array(N1)
N1 = np.array(N1)
print(N1)
N2 = []
for num in range(L):
    x = random.randint(2,10)
    N2.extend([x])
N2 = np.array(N2)
print(N2)
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
    indivCluster = np.ndarray(shape=(numIndividuals, d1)) #make a cluster list
    #for i in range(numIndividuals):
        #cluster.append([]) #at start, each individual in cluster is represented by en empty list

    for dem in range(d1): #for each demographic
        indivCluster[:,dem] = np.random.normal(M1[clusterNum][dem], S1[clusterNum][dem], numIndividuals)
        # demographicValueDistribution =  #make a distribution of values for the current demographic of the cluster. There will be one value in the generated distribution per individual in the cluster
        # i=0
        # for individual in cluster: #For each person in the cluster, fill-in the current demographic
        #     individual.extend([demographicValueDistribution[i]])
        #     i=i+1
    V[clusterNum]=indivCluster

#
# for clusterNum in range(K):
#     print("**********************************************************")
#     print("**********************************************************")
#     print("current cluster num is "+ str(clusterNum))
#     print("**********************************************************")
#     print("**********************************************************")
#
#     c = V.get(clusterNum)
#     j =0
#     for individual in c:
#         print(str(j)+"th individual has the following values for the 10 demographics: " )
#         i =0
#         j=j+1
#         for dem in individual:
#             print("value of demographic "+str(i)+" is:")
#             print(dem)
#             i=i+1


U = {}

for clusterNum in range(L): #have to populate each of the L clusters
    numProperties = N2[clusterNum] #get specified num of individuals in the current cluster
    propertyCluster = np.ndarray(shape=(numProperties, d2)) #make a cluster matrix
    for ch in range(d2): #for each demographic
        propertyCluster[:,ch] = np.random.normal(M2[clusterNum][ch], S2[clusterNum][ch], numProperties)
        print("propertyCluster is ")
        print(propertyCluster)
        U[clusterNum]=propertyCluster

for clusterNum in range(L):
    print("**********************************************************")
    print("**********************************************************")
    print("current cluster num is "+ str(clusterNum))
    print("**********************************************************")
    print("**********************************************************")

    c = U.get(clusterNum)
    j =0
    for property in c:
        print(str(j)+"th property has the following values for the 10 characteristics: " )
        i =0
        j=j+1
        for characteristic in property:
            print("value of characteristic "+str(i)+" is:")
            print(characteristic)
            i=i+1


#we have 2 matrices, V and U. we want to randomly match each entry in U to an entry in V.
#so we want to iterate through each individual from each cluster from V.
#for each of these individuals, we randomly select a property from U, the np array containing the clusters of properties
numIndividuals = np.sum(N1)
applications = []
lenderArray = np.array([0,1,2,3])
allClusters = V.keys()
print("print")
#because appending to numpy arrays is slow, will make list first then use it to construct an np.array
for clusterName in allClusters:

    cluster = V.get(clusterName)
    for individual in cluster:
        keyList = list(U.keys())
        # print(U)
        randomPropClusterInd = random.randint(0,len(keyList)-1)
        randomPropCluster = U.get(keyList[randomPropClusterInd]) #get a cluster of properties
        print("randomPropCluster is ")
        print(randomPropCluster)
        randomPropInd = random.randint(0,len(randomPropCluster)-1)
        randomProperty = randomPropCluster[randomPropInd] #replace =True means that the same property can be selected multiple times
        print(randomProperty)
        randomLender = np.random.choice(lenderArray, replace=True)
        newTup = (individual, randomProperty, randomLender)
        applications.append((newTup))


applications=np.array(applications)
for app in applications:
    print("an app is ")
    print(app)
