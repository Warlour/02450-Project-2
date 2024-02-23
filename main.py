# Python
from typing import Optional

# Reading data
from ucimlrepo import fetch_ucirepo
from functions import colorize_json
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, uci_name: Optional[str] = None, uci_id: Optional[int] = None):
        # Load dataset from uci
        self.datasetrepo = fetch_ucirepo(name=uci_name, id=uci_id)
        self.uci_name = uci_name if uci_name else self.datasetrepo.name
        self.uci_id = uci_id if uci_id else self.datasetrepo.id

        ### Headers ###
        self.X = self.datasetrepo.data.features # Attribute values (features)
        self.y = self.datasetrepo.data.targets # Class values (targets)

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

    def export_xlsx(self, filename: str = ""):
        if not filename:
            filename = f"uci_{self.uci_id}.xlsx"

        # Create DataFrame
        df = pd.DataFrame(self.X, columns=self.attributeNames)
        df['Class'] = self.y

        # Export
        df.to_excel(filename, index=True)
    
if __name__ == "__main__":
    dataset = Dataset(uci_id = 545)
    