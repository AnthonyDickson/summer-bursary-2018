# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 23:04:30 2018

@author: Fred

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import rcParams

#rcParams(fontsize=14)

#Read csv file of spectra readings
df = pd.read_csv('16_ms_lag_codes.csv')

#Make new table to turn into a np array? Without the wavelength value? 1st row is headers
#1043 rows x 47 columns ?
data_array = np.array(np.mat(df))
fig = plt.figure(num=10)
fig.clf()
plt.plot(data_array[:,0],data_array[:,1:]);

fig = plt.figure(num=11)
fig.clf()
plt.plot(data_array[:,1:]);

#biological columns
#buggies = data_array[:,1:]
buggies = data_array[216:943,1:] #remove the lower wavength

#wavelength column only
#wavelength = data_array[:,0]
wavelength = data_array[216:943,0]

#Center and scale data
scaled_data = preprocessing.scale(buggies.T)

#Make PCA plot for each organism
pca = PCA(n_components=4) # define PCA 4 seems to be enough when lower wavelength are removed
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)  # apply PCA to spectra
pca_data_bc = pca.transform(scaled_data[0:11,:])
pca_data_ec = pca.transform(scaled_data[12:19,:])
pca_data_lm = pca.transform(scaled_data[20:24,:])
pca_data_pa = pca.transform(scaled_data[25:30,:])
pca_data_se = pca.transform(scaled_data[31:35,:])
pca_data_sa = pca.transform(scaled_data[36:47,:])

#Calculate percentage of variation around the origin (mean) along row of each column
per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)

fig = plt.figure(num=1)
fig.clf()
pcvector = pca.components_ # get the PC vectors
plt.plot(wavelength,pcvector[0,:]*per_var[0],'b',label='PC1');
plt.plot(wavelength,pcvector[1,:]*per_var[1],'r--',label='PC2');
plt.plot(wavelength,pcvector[2,:]*per_var[2],'g:',label='PC3');
plt.plot(wavelength,pcvector[3,:]*per_var[3],'k-.',label='PC4');
#plt.plot(wavelength,pcvector[4,:],'mo',label='PC5');
#plt.plot(wavelength,pcvector[5,:],'k+',label='PC6');

fig = plt.figure(num=2)
fig.clf()

#Code a colour for each species so it isn't the random colour
plt.scatter(pca_data_bc[:,0], pca_data_bc[:,1]) #blue
plt.scatter(pca_data_ec[:,0], pca_data_ec[:,1]) #orange
plt.scatter(pca_data_lm[:,0], pca_data_lm[:,1]) #green
plt.scatter(pca_data_pa[:,0], pca_data_pa[:,1]) #red
plt.scatter(pca_data_se[:,0], pca_data_se[:,1]) #purple
plt.scatter(pca_data_sa[:,0], pca_data_sa[:,1]) #brown
 

fig = plt.figure(num=3)
fig.clf()
ax = fig.add_subplot(111, projection='3d')
#Code a colour for each species so it isn't the random colour
ax.scatter(pca_data_bc[:,0], pca_data_bc[:,1], pca_data_bc[:,2]) #blue
ax.scatter(pca_data_ec[:,0], pca_data_ec[:,1], pca_data_ec[:,2]) #orange
ax.scatter(pca_data_lm[:,0], pca_data_lm[:,1], pca_data_lm[:,2]) #green
ax.scatter(pca_data_pa[:,0], pca_data_pa[:,1], pca_data_pa[:,2]) #red
ax.scatter(pca_data_se[:,0], pca_data_se[:,1], pca_data_se[:,2]) #purple
ax.scatter(pca_data_sa[:,0], pca_data_sa[:,1], pca_data_sa[:,2]) #brown


fig = plt.figure(num=4)
fig.clf()
ax = fig.add_subplot(111, projection='3d')
#Code a colour for each species so it isn't the random colour
ax.scatter(pca_data_bc[:,3], pca_data_bc[:,1], pca_data_bc[:,2]) #blue
ax.scatter(pca_data_ec[:,3], pca_data_ec[:,1], pca_data_ec[:,2]) #orange
ax.scatter(pca_data_lm[:,3], pca_data_lm[:,1], pca_data_lm[:,2]) #green
ax.scatter(pca_data_pa[:,3], pca_data_pa[:,1], pca_data_pa[:,2]) #red
ax.scatter(pca_data_se[:,3], pca_data_se[:,1], pca_data_se[:,2]) #purple
ax.scatter(pca_data_sa[:,3], pca_data_sa[:,1], pca_data_sa[:,2]) #brown

plt.show()
