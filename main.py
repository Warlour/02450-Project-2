# Python
from typing import Optional, Literal

# Reading data
from ucimlrepo import fetch_ucirepo
from ucimlrepo.dotdict import dotdict # Dataset datatype
from functions import colorize_json
import numpy as np
import pandas as pd

# Set pandas display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# PCA
from scipy.linalg import svd

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

class Dataset:
    def __init__(self, uci_name: Optional[str] = None, uci_id: Optional[int] = None):
        # Load dataset from uci
        self.datasetrepo: dotdict = fetch_ucirepo(name=uci_name, id=uci_id)
        self.uci_name: str = self.datasetrepo.metadata.name
        self.uci_id: int = self.datasetrepo.metadata.uci_id

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

        self.PCA_run = False

        self.colors = ["royalblue", "orange"]

    def __str__(self):
        return self.datasetrepo.__str__()
    
    def plot_feature_compare(self, x_axis: int = 0, y_axis: int = 1):
        # Create copies
        X_c = self.X.copy()
        y_c = self.y.copy() 
        attributeNames_c = self.attributeNames.copy()

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
            plt.scatter(x_values[idx], y_values[idx], color = self.colors[i], label = self.classNames[i], edgecolors='black')

        plt.legend()
        plt.xlabel(attributeNames_c[x_axis])
        plt.ylabel(attributeNames_c[y_axis])

        plt.tight_layout()
        plt.show()

    def plot_sns_feature(self, feature: str, kind: Literal['hist', 'kde', 'ecdf'], save: bool = True):
        if feature not in self.attributeNames:
            raise ValueError(f"Invalid feature: {feature}. Available features: {', '.join(self.attributeNames)}")

        # Create copies
        X_dataframe_c = self.X_dataframe.copy()

        exclude_features = self.attributeNames.copy()

        exclude_features.remove(feature)

        # Remove excluded features
        X_dataframe_c.drop(exclude_features, axis=1, inplace=True)

        # Add Class for hue
        X_dataframe_c['Class'] = self.y_dataframe
        
        # Create Pandas DataFrame
        df = pd.DataFrame(X_dataframe_c, columns=X_dataframe_c.columns)

        p = sns.displot(data=df, x=feature, hue="Class", alpha=0.5, kind=kind, multiple='stack', bw_adjust=5)

        if save:
            p.savefig(f"{feature}_{kind}.png")

    def plot_features(self, 
                      exclude_features: Optional[list] = [], 
                      kind: Literal['scatter', 'kde', 'hist', 'reg'] = "scatter", 
                      diag_kind: Literal['auto', 'hist', 'kde'] = None,
                      plot: bool = True):
        """
        Plot all features in a pairplot.

        Parameters:
        - exclude_features: List of features to exclude from the pairplot.
        - kind: Kind of plot for the non-diagonal subplots. {['scatter'], 'kde', 'hist', 'reg'}
        - diag_kind: Kind of plot for the diagonal subplots. {'auto', 'hist', 'kde', [None]}
        - plot: Whether to plot the pairplot or not.
        """
        X_dataframe_c = self.X_dataframe.copy()

        # We need class as header for seaborn pairplot so we cannot use self.attributeNames
        attributeNames = list(self.datasetrepo.data.headers)
        for feature in exclude_features:
            attributeNames.remove(feature)

        # Remove excluded features
        X_dataframe_c.drop(exclude_features, axis=1, inplace=True)
        
        # Add class column to dataframe
        X_dataframe_c['Class'] = self.y_dataframe

        # Create Pandas DataFrame
        df = pd.DataFrame(X_dataframe_c, columns=attributeNames)

        # Create pairplot
        print("Creating pairplot with features: ", ', '.join(attributeNames), "...", sep="")
        sns.pairplot(df, hue='Class', kind=kind, diag_kind=diag_kind)

        figname = f"pairplot_{self.uci_id}_kind={kind}_diagkind={diag_kind}{f'_excluded{len(exclude_features)}' if exclude_features else ''}.png"
        print(f"Saving {figname}...")
        plt.savefig(figname)
        if plot:
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

        for patch, color in zip(bplot['boxes'], self.colors):
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

    def PCA(self):
        '''
        Perform PCA on the dataset and plot the variance explained by the principal components, aswell as the given principal components in a scatter plot.

        Parameters:
        - threshold: The threshold for the cumulative variance explained by the principal components.
        - plot: Whether to create plots.
        - save: Whether to save plots.
        - indices: The indices of the principal components to be plotted.
        '''
        print("Running PCA...")
        X_c = self.X.copy()
        X_c = X_c.astype(float)

        # Subtract mean value from data
        print("Subtracting mean value from data...")
        Y = X_c - np.ones((self.N, 1)) * X_c.mean(0)
        Y = Y * (1 / np.std(Y, 0))

        # PCA by computing SVD of Y
        print("Computing SVD...")
        U, S, Vh = svd(Y, full_matrices=False)
        self.V = Vh.T

        ### Principal components ###
        # Project the centered data onto principal component space
        self.Z = Y @ self.V

        ### Variance explained ###
        # Compute variance explained by principal components
        print("Computing variance explained by principal components...")
        self.rho = (S * S) / (S * S).sum()

        # Print all
        print("Rho:", self.rho)

        self.PCA_run = True

    def PCA_plot_PCs(self,
                     save: bool = True,
                     indices: Optional[tuple] = (0, 1)):
        '''
        Plot given principal components in a scatter plot.

        Parameters:
        - save: Whether to save the plot.
        - indices: The indices of the principal components to be plotted.
        '''
        if not self.PCA_run:
            self.PCA()

        # Copies
        y_c = self.y.copy()

        # Indices of the principal components to be plotted
        # i, j are principal components:
        i = indices[0]
        j = indices[1]

        f = plt.figure()
        plt.title("PCA of " + self.uci_name + " dataset")


        for c, cl in enumerate(self.classNames):
            # Get list of boolean values for each class
            class_mask = y_c == cl
            class_mask = class_mask.ravel() # Make 2D 1-column array into 1D array

            plt.plot(self.Z[class_mask, i], self.Z[class_mask, j], "o", color=self.colors[c], alpha=0.5, markeredgecolor='black', markeredgewidth=0.3)

        plt.legend(self.classNames)
        plt.xlabel(f"PC{i + 1}")
        plt.ylabel(f"PC{j + 1}")
        
        figname = f"PC{i+1}-{j+1}.png"
        if save:
            print(f"Saving {figname}...")
            f.savefig(figname)

    def PCA_plot_variance_explained(self,
                                    save: bool = True,
                                    threshold: float = 0.9):
        '''
        Plot variance explained by principal components.

        Parameters:
        - save: Whether to save the plot.
        - threshold: The threshold for the cumulative variance explained by the principal components.
        '''
        if not self.PCA_run:
            self.PCA()
        
        # Plot variance explained
        vp = plt.figure()
        plt.plot(range(1, len(self.rho) + 1), self.rho, "x-")
        plt.plot(range(1, len(self.rho) + 1), np.cumsum(self.rho), "o-")
        plt.plot([1, len(self.rho)], [threshold, threshold], "k--")
        plt.title("Variance explained by principal components")
        plt.xlabel("Principal component")
        plt.ylabel("Variance explained")
        plt.legend(["Individual", "Cumulative", "Threshold"])
        plt.grid()

        figname = f"variance_explained-threshold{threshold}.png"
        if save:
            print(f"Saving {figname}...")
            vp.savefig(figname)

    def PCA_pairplot(self,
                     exclude_pcs: Optional[list] = [],
                     kind: Literal['scatter', 'kde', 'hist', 'reg'] = "scatter",
                     diag_kind: Literal['auto', 'hist', 'kde'] = None,
                     save: bool = True):
        '''
        Create a pairplot of all combinations of principal components.

        Parameters:
        - kind: Kind of plot for the non-diagonal subplots. {['scatter'], 'kde', 'hist', 'reg'}
        - diag_kind: Kind of plot for the diagonal subplots. {'auto', 'hist', 'kde', [None]}
        - save: Whether to save the plot.
        '''
        if not self.PCA_run:
            self.PCA()

        Z_c = self.Z.copy()

        # Create Pandas DataFrame from PCA
        df = pd.DataFrame(Z_c, columns=[f"PC{i+1}" for i in range(Z_c.shape[1])])
        df['Class'] = self.y_dataframe

        for pc in exclude_pcs:
            if pc in df.columns:
                df.drop(pc, axis=1, inplace=True)


        print(df.head())
        # Pairplot with seaborn
        pairplot = sns.pairplot(df, hue='Class', kind=kind, diag_kind=diag_kind)

        if save:
            pairplot.savefig(f"pairplot_PCA{f'_excluded{len(exclude_pcs)}' if exclude_pcs else ''}.png")

    def PCA_plot_component_coeff(self, pcs: Optional[list] = []):
        '''
        Plot the principal component coefficients.

        Parameters:
        - pcs: The principal components to plot.
        '''
        if not self.PCA_run:
            self.PCA()
        
        legendStrs = [f"PC{i+1}" for i in range(len(pcs))]
        c = ["r", "g", "b"]
        bw = 0.2
        r = np.arange(1, self.M + 1)
        for i in pcs:
            plt.bar(r + i * bw, self.V[:, i], width=bw)
        
        # Shorten attribute names to 4 letters
        attributeNames = [name[:4] for name in self.attributeNames]
        plt.xticks(r + bw, attributeNames)
        plt.xlabel("Features")
        plt.ylabel("Coefficients")
        plt.legend(legendStrs)
        plt.grid()
        plt.title(self.uci_name + ": PCA Component Coefficients")

if __name__ == "__main__":
    dataset = Dataset(uci_id = 545)
    # Features: Area, Perimeter, Major_Axis_Length, Minor_Axis_Length, Eccentricity, Convex_Area, Extent
    # Targets: Class (Cammeo, Osmancik)

    # dataset.plot_feature_compare(0, 0)

    # dataset.plot_boxplot(feature_idx = 1)
    # dataset.plot_features(kind='kde', diag_kind = 'kde', plot = False)
    # dataset.plot_features(exclude_features = ["Extent", "Eccentricity"], kind='kde', diag_kind = 'kde', plot = False)
    # dataset.plot_features(exclude_features = ["Perimeter", "Major_Axis_Length", "Minor_Axis_Length", "Eccentricity", "Convex_Area", "Extent"], kind='kde', diag_kind = 'kde', plot = False)
    # dataset.plot_features(exclude_features = ["Area", "Perimeter", "Major_Axis_Length", "Minor_Axis_Length", "Convex_Area"], kind='kde', diag_kind = 'kde', plot = False)
    # dataset.plot_sns_feature("Area", kind="kde", save=False)
    # dataset.plot_sns_feature("Extent", kind="kde", save=False)
    # dataset.plot_sns_feature("Convex_Area", kind="kde", save=False)
    dataset.PCA_plot_component_coeff(pcs=[0, 1, 2])

    # dataset.PCA()


    # dataset.PCA_plot_variance_explained(threshold=0.5)
    # dataset.PCA_plot_PCs(indices=(2, 3))
    # dataset.PCA_pairplot()
    # dataset.PCA_pairplot(exclude_pcs=["PC4", "PC5", "PC6", "PC7"])
    plt.show()