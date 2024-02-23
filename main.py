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
    
    def plot_features(self, x_axis: int = 0, y_axis: int = 1):
        # Make a simple plot of the i'th attribute against the j'th attribute
        plt.plot(self.X[:, x_axis], self.X[:, y_axis], "o")
        plt.title(f"{self.uci_name} dataset")
        plt.xlabel(self.attributeNames[x_axis])
        plt.ylabel(self.attributeNames[y_axis])
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
    # print(dataset.datasetrepo.metadata.uci_id)