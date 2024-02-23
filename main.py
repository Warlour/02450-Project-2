# Reading data
from ucimlrepo import fetch_ucirepo
from functions import colorize_json
from numpy import unique

# Plotting
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, uci_repo_id: int):
        # Load dataset from uci
        self.datasetrepo = fetch_ucirepo(id=uci_repo_id)

        ### Headers ###
        self.X = self.datasetrepo.data.features # Attribute values (features)
        self.y = self.datasetrepo.data.targets # Class values (targets)

        ### Headers ###
        self.attributeNames = list(self.datasetrepo.data.headers)
        self.attributeNames.remove('Class') # Remove 
        self.classNames = unique(y)

        ### Data lengths ###
        self.N = len(self.y)
        self.M = len(self.attributeNames)
        self.C = len(self.classNames)

    def __str__(self):
        return self.datasetrepo.__str__()