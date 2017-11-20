__author__ = 'smaillot'

from sklearn.decomposition import PCA
import numpy as np
from pandas.tools.plotting import scatter_matrix
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt


def compute_pca(X):
    df = pd.DataFrame(X)
      
    #scatter = scatter_matrix(df, alpha = 0.2, figsize=(6, 6), diagonal='kde')
    pca = PCA()
    pca.fit(X)
    pca_var = pca.explained_variance_ratio_
    #plt.plot(range(1,len(X[0])+1),np.cumsum(pca_var))
    #plt.plot([1, len(X[0])+1],[0.95, 0.95])
    #plt.xlabel = "Principal Component"
    #plt.ylabel = "Cumulative Proportion of Variance Explained"
    
    npc = 10
    X_reduced = pca.fit_transform(X)
    X_reduced = X_reduced[:,0:10]
    df = pd.DataFrame(X_reduced)
    #scatter = scatter_matrix(df, alpha = 0.2, diagonal='kde')
    return pca