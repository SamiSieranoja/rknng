#!/usr/bin/python3

import numpy as np
import rpdivknng



# Class needs to have following properties:
# - attribute 'size' that reflects the number of data objects
# - function distance(a,b) where parameters a,b are integers between 0..(size-1)
# Name of class is arbitrary
class DistanceMeasureL2:
	def __init__(self,x):
		self.tmp = 1
		self.x = x
		self.size = len(x)
	def distance(self,a,b):
		dist = 3.3
		dist = np.linalg.norm(self.x[a]-self.x[b])
		return dist

x=np.loadtxt('data/g2-256-50.txt')
dist = DistanceMeasureL2(x)
# knn = rpdivknng.rpdiv_knng(x,20,window=30,nndes=0.2,maxiter=100,delta=0.02)

knn = rpdivknng.rpdiv_knng_o(dist,20)
# knn = rpdivknng.rpdiv_knng(x,20,window=30,nndes=0.2,maxiter=100,delta=0.02)
# knn = rpdivknng.rpdiv_knng_o(dist,20,delta=0.02,nndes=0.0)
# print(knn)
print("KNN:")
# print(knn[0])
# print(knn[1])

