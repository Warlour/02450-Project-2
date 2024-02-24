import pandas as pd
import numpy as np

Cammeo_label = 'Cammeo'
Osmancik_label = 'Osmancik'
dec = 4

X = pd.read_csv("Rice_dataset.csv")
X = X.to_numpy()

labels = X[:, -1]

mask1 = (labels == Cammeo_label)
mask2 = ~mask1

Cam_X = X[mask1]
Osm_X = X[mask2]

print("Cammeo")
for i in range(1,X.shape[1]-1):
    temp = np.quantile(Cam_X[:,i],[0,.25,.5,.75,1]).astype(float)
    temp = np.round(temp,dec)
    print(temp,"\nSTD: {}, Mean: {}"
    .format(np.round(np.std(Cam_X[:,i]),dec),np.round(np.mean(Cam_X[:,i]),dec)))
print("Osmancik")
for i in range(1,X.shape[1]-1):
    temp = np.quantile(Osm_X[:,i],[0,.25,.5,.75,1]).astype(float)
    temp = np.round(temp,dec)
    print(temp,"\nSTD: {}, Mean: {}"
    .format(np.round(np.std(Osm_X[:,i]),dec),np.round(np.mean(Osm_X[:,i]),dec)))

