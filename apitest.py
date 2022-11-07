#!/usr/bin/python3

import os
import time
import numpy as np

import spam

print("LOGIT:")
x=np.loadtxt('data/g2-256-50.txt')
# print(x[0])
#print(spam.logit(0.1))
# knn = spam.rpdiv_knng(x,20,30)
# knn = spam.rpdiv_knng(x,20,30,0.2,100,delta=0.02)
# knn = spam.rpdiv_knng(x,20,30,0.2,100)

# knn = spam.rpdiv_knng(x,20,window=30,nndes=0.2,maxiter=100,delta=0.02)
knn = spam.rpdiv_knng(x,20)
print("KNN:")
print(knn[0])
print(knn[1])
#print(spam.logit([1,2,3]))


