import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 12:24:15 2022

@author: Aaron-PC
"""

"""
Project2-AaronMcGuirk.py

Text clustering HW#2 for CSIS320

@author: Aaron McGuirk
@version: Spring 2022
"""

"""
Notes:
    School of Liberal Arts = 1
    School of Science = 2
    School of Business = 3
"""

"""
def clustering_33_groups():
    

    
def clustering_57_groups():
    

def clustering_optimal_groups():
"""

data = np.loadtxt("descriptions.txt", dtype="str", delimiter="\t", skiprows=1)
y_km = data


classes_vec = CountVectorizer().fit(data)
print(classes_vec.vocabulary_)

def clustering_3_groups():
    
    #km = KMeans(n_clusters=3)
    #km.fit(data)
    #y_km = km.predict(data)
    
    #plt.scatter(data[:, 0], data[:, 1], c=y_km, s=50, cmap='viridis')
    #centers = km.cluster_centers_
    #plt.scatter(centers[:, 0], centers[:, 1], c='red', s=1314, alpha=0.5)
    
    ag = AgglomerativeClustering(n_clusters = 3).fit(data)
    centers = avg_clusters(data,ag.labels_)
    
    
    #for i in range(len(data)):
        
def avg_clusters(data,clusters):
    groups = np.unique(clusters)
    nrows = len(groups)
    ncols = data.shape[1]
    result = np.empty((nrows,ncols))
    for g in groups:
        x = data[clusters==g]
        for c in range(ncols):
            result[g,c] = np.average(x[:,c])
    return result
        
        
