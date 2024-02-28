# Python
from typing import Optional, Literal

# Reading data
from ucimlrepo import fetch_ucirepo
from ucimlrepo.dotdict import dotdict # Dataset datatype
from functions import *
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
    
    def plot_feature_compare(self, x_axis: int = 0, y_axis: int = 1, save: bool = True):
        '''
        Plot two features against each other.

        Parameters:
        - x_axis: The index of the feature to plot on the x-axis.
        - y_axis: The index of the feature to plot on the y-axis.
        - save: Whether to save the plot.
        '''
        # Get single column containing the x and y values
        x_values = self.X[:, x_axis]
        y_values = self.X[:, y_axis]

        # Create subplot
        plt.figure(figsize=(10, 5))
        # rows, columns, index
        plt.subplot(1, 1, 1)
        plt.title(f"{self.uci_name} dataset")

        for i, cl in enumerate(self.classNames):
            # Get list of boolean values for each class
            idx = self.y == cl
            idx = idx.ravel() # Make 2D 1-column array into 1D array

            # Create scatter points
            plt.scatter(x_values[idx], y_values[idx], color = self.colors[i], label = self.classNames[i], edgecolors='black', alpha=0.5, linewidths=0.5)

        plt.legend()
        plt.xlabel(self.attributeNames[x_axis])
        plt.ylabel(self.attributeNames[y_axis])

        plt.tight_layout()
        if save:
            plt.savefig(f"feature_compare_{self.attributeNames[x_axis]}-{self.attributeNames[y_axis]}.png")

    def plot_sns_feature(self, feature: str, kind: Literal['hist', 'kde', 'ecdf'], save: bool = True):
        '''
        Plot a single feature with seaborn library. Useful for plotting the distribution of a feature.

        Parameters:
        - feature: The feature to plot.
        - kind: The kind of plot to create. {['hist'], 'kde', 'ecdf'}
        - save: Whether to save the plot.
        '''
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

        if kind == "kde":
            p = sns.displot(data=df, x=feature, hue="Class", alpha=0.5, kind=kind, multiple='stack', bw_adjust=5)
            plt.title(f"Kernel Density Estimation of {feature}")
        elif kind == "hist":
            p = sns.displot(data=df, x=feature, hue="Class", alpha=0.5, kind=kind, multiple='stack')
            plt.title(f"Histogram of {feature}")
        elif kind == "ecdf":
            p = sns.displot(data=df, x=feature, hue="Class", alpha=0.5, kind=kind)
            plt.title(f"Empirical Cumulative Distribution Function of {feature}")

        plt.tight_layout()
        if save:
            p.savefig(f"{feature}_{kind}.png")

    def plot_features(self, 
                      features: Optional[list] = [], 
                      kind: Literal['scatter', 'kde', 'hist', 'reg'] = "scatter", 
                      diag_kind: Literal['auto', 'hist', 'kde', None] = None,
                      save: bool = True):
        """
        Plot all features in a pairplot.

        Parameters:
        - features: List of features to include from the pairplot.
        - kind: Kind of plot for the non-diagonal subplots. {['scatter'], 'kde', 'hist', 'reg'}
        - diag_kind: Kind of plot for the diagonal subplots. {'auto', 'hist', 'kde', [None]}
        - plot: Whether to plot the pairplot or not.
        """
        if not features:
            features = self.attributeNames
        
        X_dataframe_c = self.X_dataframe.copy()

        # We need class as header for seaborn pairplot so we cannot use self.attributeNames
        attributeNames = list(self.datasetrepo.data.headers)

        exclude_features = self.attributeNames.copy()
        for feature in features:
            exclude_features.remove(feature)

        for feature in exclude_features:
            attributeNames.remove(feature)

        # Remove excluded features
        X_dataframe_c.drop(exclude_features, axis=1, inplace=True)
        
        # Add class column to dataframe
        X_dataframe_c['Class'] = self.y_dataframe

        # Create Pandas DataFrame
        df = pd.DataFrame(X_dataframe_c, columns=attributeNames)

        # Create pairplot
        print("Creating pairplot with features: ", ', '.join(features), "...", sep="")
        # plt.rcParams['axes.labelsize'] = 50
        sns.set_context("paper", font_scale=1.5)
        p = sns.pairplot(df, hue='Class', kind=kind, diag_kind=diag_kind)
        # p.tick_params(axis='both', which="minor", labelsize=50)

        if features == self.attributeNames:
            included: str = "all"
        else:
            included: str = ', '.join(features)
        figname = f"pairplot_{self.uci_id}_kind={kind}_diagkind={diag_kind}_{f'incl={included}' if features else 'all'}.png"
        # plt.tight_layout()
        if save:
            print(f"Saving {figname}...")
            plt.savefig(figname)

    def plot_boxplot(self, feature_idx: int = 0, save: bool = True):
        """
        Plots boxplots for each class next to each other on a single axis.

        Parameters:
        - feature_idx: The index of the feature to plot.
        """

        boxplot_data = []
        values = self.X[:, feature_idx]

        for i, cl in enumerate(self.classNames):
            # Get list of boolean values for each class
            idx = self.y == cl
            idx = idx.ravel() # Make  2D  1-column array into  1D array

            # Add data for class to boxplot_data list
            boxplot_data.append(values[idx])
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 8))
        fontsize = 20

        # Plot boxplots for each class next to each other on same axis
        bplot = ax.boxplot(boxplot_data, 
                           patch_artist = True, 
                           notch = True, 
                           vert = True, 
                           meanline = False, 
                           showmeans = False, 
                           showcaps = True, 
                           showbox = True, 
                           showfliers = True, 
                           capwidths=[0.3, 0.3],
                           widths = 0.6)
        
        # Set title and labels
        # ax.set_title(f"Boxplot for {self.attributeNames[feature_idx]} of classes", fontsize=fontsize)

        # ax.set_xlabel("Class", fontsize=fontsize)
        ax.set_ylabel(f"{self.attributeNames[feature_idx]}", fontsize=fontsize)

        ax.set_xticks(range(1, len(self.classNames) + 1))
        ax.set_xticklabels(self.classNames, fontsize=fontsize)

        # ax.tick_params(axis='both', which='major', labelsize=16)

        for patch, color in zip(bplot['boxes'], self.colors):
            patch.set_facecolor(color)

        plt.tight_layout()
        if save:
            plt.savefig(f"boxplot_{self.attributeNames[feature_idx]}.png")

    def export_xlsx(self, filename: str = ""):
        '''
        Export dataset to an Excel file.

        Parameters:
        - filename: The name of the file to export to. Default is "uci_" + dataset name.
        '''
        if not filename:
            filename = f"uci_{self.uci_id}.xlsx"

        # Create DataFrame
        df = pd.DataFrame(self.X_dataframe, columns=self.attributeNames)
        df['Class'] = self.y_dataframe

        # Export
        df.to_excel(filename, index=True)

    def PCA(self, save_basis: bool = True):
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
        self.S = S

        ### Principal components ###
        # Project the centered data onto principal component space
        self.Z = Y @ self.V
        if save_basis:
            np.savetxt(f"PCA_basis_{self.uci_id}.csv", self.V, delimiter=",")

            # for i in range(self.Z.shape[0]):
            # print(self.Z)

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

        # Indices of the principal components to be plotted
        # i, j are principal components:
        i = indices[0]
        j = indices[1]

        f = plt.figure()
        plt.title("PCA of " + self.uci_name + " dataset")


        for c, cl in enumerate(self.classNames):
            # Get list of boolean values for each class
            class_mask = self.y == cl
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
        plt.tight_layout()

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

        # Create Pandas DataFrame from PCA
        df = pd.DataFrame(self.Z, columns=[f"PC{i+1}" for i in range(self.Z.shape[1])])
        df['Class'] = self.y_dataframe

        for pc in exclude_pcs:
            if pc in df.columns:
                df.drop(pc, axis=1, inplace=True)


        print("Creating PCA pairplot...")
        # Pairplot with seaborn
        pairplot = sns.pairplot(df, hue='Class', kind=kind, diag_kind=diag_kind,)
        plt.rcParams['axes.labelsize'] = 50
        pairplot.tight_layout()

        if save:
            pairplot.savefig(f"pairplot_PCA{f'_excluded{len(exclude_pcs)}' if exclude_pcs else ''}.png")

    def PCA_plot_component_coeff(self, pcs: Optional[list] = [], save: bool = True):
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
        plt.tight_layout()

        if save:
            print(f"Saving PCA_component_coefficients.png...")
            plt.savefig(f"PCA_component_coefficients.png")

if __name__ == "__main__":
    dataset = Dataset(uci_id = 545)
    # Features: Area, Perimeter, Major_Axis_Length, Minor_Axis_Length, Eccentricity, Convex_Area, Extent
    # Targets: Class (Cammeo, Osmancik)

    # dataset.plot_feature_compare(0, 1, save=False)
    # dataset.plot_sns_feature("Area", 'kde', save=True)
    # dataset.plot_features(features=["Area", "Perimeter", "Major_Axis_Length", "Minor_Axis_Length"], kind="scatter", diag_kind="kde", save=False)
    dataset.plot_features(kind="scatter", diag_kind="kde", save=True)
    # dataset.plot_boxplot(0, save=True)
    # dataset.export_xlsx()

    # dataset.PCA()
    # print("\n Cum sum", np.cumsum(dataset.rho))
    # print("\n Length of rho", len(dataset.rho))
    # print("\n V", dataset.V)
    # print("\n S", dataset.S)
    # dataset.PCA_plot_PCs(save = False, indices=(0, 1))
    # dataset.PCA_plot_variance_explained(save = False, threshold=0.9)
    # dataset.PCA_pairplot(exclude_pcs=[f"PC{i+1}" for i in range(4, 6)], kind="scatter", diag_kind="kde", save=True)
    # dataset.PCA_pairplot(kind="scatter", diag_kind="kde", save=True)
    # dataset.PCA_plot_component_coeff(pcs=[0, 1, 2, 3], save = False)

    # Use plt.show to plot
    # plt.show()