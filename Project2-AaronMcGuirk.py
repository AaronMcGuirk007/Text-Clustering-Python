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

num_schools = 3
schools = ["School of Liberal Arts", "School of Science", "School of Business"]

num_departments = 33

num_prefixes = 57

num_optimize = 0

data = np.loadtxt("descriptions.txt", dtype="str", delimiter="\t", skiprows=1)
data_vec = TfidVectorizer("descriptions.txt")
#y_km = data
#TBD = 1


def numOfSchoolsClustering(data, clusters):
    
    km = kMeans(n_clusters=clusters, solver='kmeans', max_df=100, learning_rate = 1)
    km.fit(data)
    centers = km.cluster_centers.argssort()
    school_labels = vectorizer.get_feature_names()
    
    #y_km = km.predict(data)
   
    #plt.scatter(data[:, 0], data[:, 1], c=y_km, s=50, cmap='viridis')
    #centers = km.cluster_centers_
    #plt.scatter(centers[:, 0], centers[:, 1], c='red', s=1314, alpha=0.5)
   
    #ag = AgglomerativeClustering(n_clusters = 3).fit(data)
    #centers = avg_clusters(data,ag.labels_)
    
    print("Number of Schools Cluster: ")
    
    for i in range(len(schools)):
        print(schools[i] + ": ")
        for j in centers[i, :num_schools]:
            print(centers)[j]
        
        
def numOfDepartmentsClustering(data, clusters):
    
    km = kMeans(n_clusters=clusters, solver='kmeans', max_df=100, learning_rate = 1)
    km.fit(data)
    centers = km.cluster_centers.argssort()
    school_labels = vectorizer.get_feature_names()
    
    #y_km2 = km2.predict(data)
    
    #plt.scatter(data[:, 0], data[:, 1], c=y_km2, s=50, cmap='viridis')
    #centers2 = km2.cluster_centers_
    #plt.scatter(centers2[:, 0], centers2[:, 1], c='red', s=1314, alpha=0.5)
   
    #ag2 = AgglomerativeClustering(n_clusters = 33).fit(data)
    #centers2 = avg_clusters(data,ag2.labels_)
    
    print("Number of Departments Cluster: ")
    
    for i in range(num_department):
        print("Dept. Cluster")
        for j in centers[i, :num_departments]:
            print(centers[j])


def numOfPrefixsClustering(data, clusters):
    
    km = kMeans(n_clusters=clusters, solver='kmeans', max_df=100, learning_rate = 1)
    km.fit(data)
    centers = km.cluster_centers.argssort()
    school_labels = vectorizer.get_feature_names()
    
    #y_km3= km3.predict(data)
   
    #plt.scatter(data[:, 0], data[:, 1], c=y_km3, s=50, cmap='viridis')
    #center3 = km3.cluster_centers_
    #plt.scatter(centers3[:, 0], centers3[:, 1], c='red', s=1314, alpha=0.5)
   
    #ag3 = AgglomerativeClustering(n_clusters = 57).fit(data)
    #centers3 = avg_clusters(data,ag3.labels_)
    
    print("Number of Prefixes Cluster: ")
    
    for i in range(num_prefixes):
        print("Prefix Cluster")
        for j in centers[i, :num_prefixes]:
            print(centers[j])
    
    
def numOptimalClustering(data, clusters):
    
    km = kMeans(n_clusters=clusters, solver='kmeans', max_df=100, learning_rate = 1)
    km.fit(data)
    centers = km.cluster_centers.argssort()
    school_labels = vectorizer.get_feature_names()
    
    #y_kmOpt = kmOpt.predict(data)
    
    #plt.scatter(data[:, 0], data[:, 1], c=y_kmOpt, s=50, cmap='viridis')
    #centers4 = kmOpt.cluster_centers_
    #plt.scatter(centers4[:, 0], centers4[:, 1], c='red', s=1314, alpha=0.5)
    
    #ag4 = AgglomerativeClustering(n_clusters = TBD).fit(data)
    #centers4 = avg_clusters(data,ag4.labels_)
    
    print("Number of Prefixes Cluster: ")
    
    for i in range(num_optimize):
        print("Prefix Cluster")
        for j in centers[i, :num_optimize]:
            print(centers[j])

        
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
        
        
def test_methods():
    
    # Tests all of the clustering methods!
    print(numOfSchoolsClustering(data, num_schools))
    
    print(numOfDepartmentsClustering(data, num_departments))
    
    print(numOfPrefixsClustering(data, num_prefixes))
    
    print(numOptimalClustering(data, num_optimize))