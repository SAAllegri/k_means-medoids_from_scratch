#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from PIL import Image
import time

class K_Means:
    def __init__(self, k=3, max_iter=50, threshold=0.001, distance_method='L2', k_medoids=False, dummy_centers=False):
        
        self.k = k
        self.max_iter = max_iter
        self.threshold = threshold
        self.distance_method = distance_method
        self.k_medoids = k_medoids
        self.dummy_centers = dummy_centers
        self.iterations = 0
        self.ssd = 0
        self.time = 0
        self.error = 0
        
    def loadData(self, np_array):

        self.np_data = np_array
        self.shape = np_array.shape
        self.dimensions = np_array.shape[1]
        self.centers = np.zeros((self.k, np_array.shape[1]))
        self.labels = np.zeros(np_array.shape[0])
        self.temp_distance_array = np.zeros((np_array.shape[0], self.k))
        self.labeled_np_data = np.zeros((np_array.shape[0], self.dimensions + 1))
        
    def distance(self, a, b, axis_val=1):
        
        if self.distance_method == 'L1':
            return np.linalg.norm(a - b, axis=axis_val, ord=1)
        
        elif self.distance_method == 'L2':
            return np.linalg.norm(a - b, axis=axis_val, ord=2)
        
        elif self.distance_method == 'inf':
            return np.linalg.norm(a - b, axis=axis_val, ord=np.inf)
        
        elif self.distance_method == '-inf':
            return np.linalg.norm(a - b, axis=axis_val, ord=-np.inf)
        
        else:
            raise('Try different norm')
    
    def initializeCenters(self):
        
        if self.dummy_centers:
            for i in range(self.k):
                ones_array = np.ones(self.dimensions)
                self.centers[i] = i * ones_array
        else:
            for i in range(self.k):
                self.centers[i] = (self.np_data[np.random.randint(0, self.shape[0])])
    
    def chooseCenters(self):

        for i in range(self.k):
            self.temp_distance_array[:, i] = self.distance(self.np_data, self.centers[i])

        self.labels = np.argmin(self.temp_distance_array, axis=1)
        self.labeled_np_data[:, :-1] = self.np_data
        self.labeled_np_data[:, -1] = self.labels
    
    def generateNewCenters(self):
        
        if self.k_medoids: 
            '''
            TOO SLOW!
            for i in range(self.k):
                subset_data = self.labeled_np_data[self.labeled_np_data[:, -1] == i][:, :self.dimensions]
    
                distance_list = np.zeros(self.shape[0])
                
                for j, dp in enumerate(subset_data):
                    distance_list[j] = np.linalg.norm(subset_data - subset_data[j])
                        
                print(distance_list)
            '''
            for i in range(self.k):
                subset_data = self.labeled_np_data[self.labeled_np_data[:, -1] == i][:, :self.dimensions]
                centroid = np.average(subset_data, axis=0)

                try:
                    self.centers[i] = subset_data[np.argmin(self.distance(subset_data, centroid, axis_val=1))]
                except:
                    # In case of empty clusters --> restart the algorithm with new initial centroids
                    self.initializeCenters()
                    self.max_iter = self.max_iter + self.iterations
                    self.iterations = -1
                    self.time = 0
                    self.error = 1
        
        else:
            for i in range(self.k):
                self.centers[i] = np.average(self.labeled_np_data[self.labeled_np_data[:, -1] == i][:, :self.dimensions], axis=0)
        
    def evaluate(self):
        
        sum_squared_distances = []
        
        for i in range(self.k):
            subset_data = self.labeled_np_data[self.labeled_np_data[:, -1] == i][:, :self.dimensions]
            subset_centroid = np.average(subset_data, axis=0)
            
            sum_squared_distances.append(np.linalg.norm(subset_data - subset_centroid) ** 2)

        self.ssd = np.sum(sum_squared_distances[0])
        
    def train(self):
        
        self.initializeCenters()
        
        old_euc_distance = 0
        
        for i in range(self.max_iter):
            
            start = time.time()
            
            old_centers = self.centers.copy()
            
            self.chooseCenters()
            self.generateNewCenters()
            
            new_centers = self.centers.copy()
            
            self.iterations = self.iterations + 1
            self.evaluate()
            
            euc_distance = np.linalg.norm(old_centers - new_centers)
            
            self.time = self.time + (time.time() - start)

            if (np.abs(old_euc_distance - euc_distance)) < self.threshold:

                break

            old_euc_distance = euc_distance