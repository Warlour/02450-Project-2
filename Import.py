import pandas as pd
import numpy as np
#path
filename = "data/Rice_dataset.csv"
#pandas dataframe
df = pd.read_csv(filename)
#exclude header:
raw_data = df.values

#exclude index and labels
cols = range(1, 8)
X = raw_data[:, cols]

#Attributes/Features:
attributeNames = np.asarray(df.columns[cols])
#Labels
classLabels = raw_data[:, -1]
classNames = np.unique(classLabels)
classDict = dict(zip(classNames, range(len(classNames))))
#List of labels converted to int
y = np.array([classDict[cl] for cl in classLabels])

#Object type not supported by SVD, convert to float:
X = X.astype(float)
C = len(classNames)
N, M = X.shape