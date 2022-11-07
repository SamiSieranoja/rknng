#!/usr/bin/python3

import os
import time
import numpy as np

import spam

print("LOGIT:")
x=np.loadtxt('data/g2-256-50.txt')
# print(x[0])
#print(spam.logit(0.1))
knn = spam.logit(x)
print("KNN:")
print(knn[0])
print(knn[1])
#print(spam.logit([1,2,3]))


