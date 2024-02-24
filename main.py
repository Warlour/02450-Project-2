# Python
from typing import Optional

# Reading data
from ucimlrepo import fetch_ucirepo
from functions import colorize_json
import numpy as np
import pandas as pd

# Set pandas display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Plotting
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, uci_name: Optional[str] = None, uci_id: Optional[int] = None):
        # Load dataset from uci
        self.datasetrepo = fetch_ucirepo(name=uci_name, id=uci_id)
        self.uci_name = self.datasetrepo.metadata.name
        self.uci_id = self.datasetrepo.metadata.uci_id

        ### Data ###
        self.X_dataframe = self.datasetrepo.data.features # Attribute values (features)
        self.y_dataframe = self.datasetrepo.data.targets # Class values (targets)

        self.X = np.array(self.X_dataframe)
        self.y = np.array(self.y_dataframe)

        ### Headers ###
        self.attributeNames = list(self.datasetrepo.data.headers)
        self.attributeNames.remove('Class') # Remove 
        self.classNames = np.unique(self.y)

        ### Data lengths ###
        self.N = len(self.y)
        self.M = len(self.attributeNames)
        self.C = len(self.classNames)

    def __str__(self):
        return self.datasetrepo.__str__()
    
    def plot_feature_compare(self, x_axis: int = 0, y_axis: int = 1):
        # Create copies
        X_c = self.X.copy()
        y_c = self.y.copy() 
        attributeNames_c = self.attributeNames.copy()

        # Colors for each class
        color = ["royalblue", "orange"]

        # Get single column containing the x and y values
        x_values = X_c[:, x_axis]
        y_values = X_c[:, y_axis]

        # Create subplot
        plt.figure(figsize=(10, 5))
        # rows, columns, index
        plt.subplot(1, 1, 1)
        plt.title(f"{self.uci_name} dataset")

        for c, cl in enumerate(self.classNames):
            # Get list of boolean values for each class
            idx = y_c == cl
            idx = idx.ravel() # Make 2D 1-column array into 1D array

            # Create scatter plot
            plt.scatter(x_values[idx], y_values[idx], color = color[c], label = self.classNames[c], edgecolors='black')

        plt.legend()
        plt.xlabel(attributeNames_c[x_axis])
        plt.ylabel(attributeNames_c[y_axis])

        plt.tight_layout()
        plt.show()

    def export_xlsx(self, filename: str = ""):
        if not filename:
            filename = f"uci_{self.uci_id}.xlsx"

        # Create DataFrame
        df = pd.DataFrame(self.X_dataframe, columns=self.attributeNames)
        df['Class'] = self.y_dataframe

        # Export
        df.to_excel(filename, index=True)
    
if __name__ == "__main__":
    dataset = Dataset(uci_id = 545)
    dataset.plot_features(x_axis = 2, y_axis = 3)
    