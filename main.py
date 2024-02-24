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
import itertools

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

        for i, cl in enumerate(self.classNames):
            # Get list of boolean values for each class
            idx = y_c == cl
            idx = idx.ravel() # Make 2D 1-column array into 1D array

            # Create scatter points
            plt.scatter(x_values[idx], y_values[idx], color = color[i], label = self.classNames[i], edgecolors='black')

        plt.legend()
        plt.xlabel(attributeNames_c[x_axis])
        plt.ylabel(attributeNames_c[y_axis])

        plt.tight_layout()
        plt.show()

    def plot_features(self): # TODO: Not done!!!
        # Create copies
        X_c = self.X.copy()
        y_c = self.y.copy() 
        attributeNames_c = self.attributeNames.copy()

        # Colors for each class
        color = ["royalblue", "orange"]

        num_features = self.M

        # Calculate rows and columns for grid
        num_rows = num_features //  2 if num_features %  2 ==  0 else num_features //  2 +  1
        num_cols =  2

        # Create figure with grid of subplots
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 5 * num_rows))
        fig.suptitle(f"{self.uci_name} dataset")

        # Flatten arrays with one column
        print(axs, "\n")

        print(axs.shape)
        axs = axs.flatten()
        print(axs)

        feature_pairs = list(itertools.product(range(num_features), repeat=2)) # 7 x 7 = 49 pairs (maybe we should take some away)
        # print(feature_pairs)

        for i, (x_axis, y_axis) in enumerate(feature_pairs):
            # print(i)
            row = i // num_cols # Integer division
            col = i % num_cols # Remainder

            # Get single column containing the x and y values
            x_values = X_c[:, x_axis]
            y_values = X_c[:, y_axis]

            # Select current subplot
            ax = axs[row, col]

            for c, cl in enumerate(self.classNames):
                # Get list of boolean values for each class
                idx = y_c == cl
                idx = idx.ravel() # Make 2D 1-column array into 1D array

                # Create scatter points
                ax.scatter(x_values[idx], y_values[idx], color = color[c], label=self.classNames[c], edgecolors='black')
            
            ax.set_xlabel(attributeNames_c[x_axis])
            ax.set_ylabel(attributeNames_c[y_axis])
            ax.legend()
        
        # Remove unused subplots
        for i in range(len(feature_pairs), num_rows * num_cols):
            row = i // num_cols
            col = i % num_cols
            fig.delaxes(axs[row, col])

        plt.tight_layout()
        plt.show()

    def plot_boxplot(self, feature_idx: int = 0):
        """
        Plots boxplots for each class next to each other on a single axis.

        Parameters:
        - feature_idx: The index of the feature to plot.
        """

        X_c = self.X.copy()
        y_c = self.y.copy()
        attributeNames_c = self.attributeNames.copy()

        label = attributeNames_c[feature_idx]

        boxplot_data = []
        values = X_c[:, feature_idx]

        colors = ["royalblue", "orange"]

        for i, cl in enumerate(self.classNames):
            # Get list of boolean values for each class
            idx = y_c == cl
            idx = idx.ravel() # Make  2D  1-column array into  1D array

            # Add data for class to boxplot_data list
            boxplot_data.append(values[idx])
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot boxplots for each class next to each other on same axis
        bplot = ax.boxplot(boxplot_data, patch_artist = True, notch = True, vert = True)
        
        # Set title and labels
        ax.set_title(f"Boxplot for {label} of classes")

        ax.set_xlabel("Class")
        ax.set_ylabel(f"{attributeNames_c[feature_idx]}")

        ax.set_xticks(range(1, len(self.classNames) + 1))
        ax.set_xticklabels(self.classNames)

        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

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
    # Features: Area, Perimeter, Major_Axis_Length, Minor_Axis_Length, Eccentricity, Convex_Area, Extent
    # Targets: Class (Cammeo, Osmancik)

    # dataset.plot_feature_compare(2, 3)
    # dataset.plot_features()
    dataset.plot_boxplot(feature_idx = 1)
    