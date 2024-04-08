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

# Decision tree
from sklearn import tree

# Logistic Regression - Rasmus

from typing import Optional, List, Tuple
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.preprocessing import LabelEncoder

import warnings
from sklearn.exceptions import ConvergenceWarning
import scipy.stats as st
import itertools

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def namestr(obj):
    return [name for name in globals() if globals()[name] is obj][0]

def compare_regression_models(y_true, y_pred_model1, y_pred_model2, alpha=0.05):
    n = len(y_true)
    # Compute differences in loss
    z = np.abs(y_true - y_pred_model1) ** 2 - np.abs(y_true - y_pred_model2) ** 2
    # Compute mean of differences
    z_hat = np.mean(z)
    # Compute standard deviation of differences
    z_tilde_sigma_sq = np.sum((z - z_hat) ** 2) / (n * (n - 1))
    z_tilde_sigma = np.sqrt(z_tilde_sigma_sq)
    # Compute confidence interval
    z_L, z_U = st.t.interval(1 - alpha, df=n - 1, loc=z_hat, scale=z_tilde_sigma)
    # Compute p-value
    p = 2 * st.t.cdf(-np.abs(z_hat) / z_tilde_sigma, df=n - 1)
    return z_L, z_U, p

def two_step_cross_validation(X, y, M: List[BaseEstimator], K1: int, K2: int, model_amounts: int, classify: bool = False) -> None:
    '''
    param X: The feature matrix.
    param y: The target vector.
    param M: List of models to be evaluated.
    param K1: Number of outer folds
    param K2: Number of inner folds
    '''
    global inputs

    print("Running two-step cross-validation...")
    output_dict = {}
    output_dict["optimal_lambdas"] = []
    output_dict["optimal_hidden_layers"] = []

    outer_cv = KFold(n_splits=K1, shuffle=True)
    inner_cv = KFold(n_splits=K2, shuffle=True)

    E_val_j = np.zeros((K2, len(M)))


    E_test_i_model1 = np.array([])
    E_test_i_model2 = np.array([])
    E_test_i_model3 = np.array([])

    for K1_i, (D_par_i, D_test_i) in enumerate(outer_cv.split(X), start=1):
        print("----------------------")
        E_gen_s = []
        X_par_i, X_test_i = X[D_par_i], X[D_test_i]
        y_par_i, y_test_i = y[D_par_i], y[D_test_i]

        for K2_j, (D_train_j, D_val_j) in enumerate(inner_cv.split(X_par_i), start=1):
            X_train_j, X_val_j = X[D_train_j], X[D_val_j]
            y_train_j, y_val_j = y[D_train_j], y[D_val_j]

            for s, model in enumerate(M, start=0):
                model.fit(X_train_j, y_train_j)
                if classify:
                    E_val_j[K2_j-1, s] = sum([a != b for a, b in zip(y_val_j, model.predict(X_val_j))]) / len(y_val_j)
                else:
                    E_val_j[K2_j-1, s] = np.square(y_val_j - model.predict(X_val_j)).sum() / y_val_j.shape[0]

        # Calculate E_gen_s for each model
        for s in range(len(M)):
            summ = sum((len(D_val_j) / len(D_par_i)) * E_val_j[j, s] for j in range(K2))
            E_gen_s.append(summ)

        


        oridx = np.argmin(E_gen_s[0:model_amounts])
        optimal_model1 = M[oridx]
        output_dict["optimal_lambdas"].append(inputs[oridx])
        print(optimal_model1)

        omidx = np.argmin(E_gen_s[model_amounts:2*model_amounts])+model_amounts
        optimal_model2 = M[omidx]
        output_dict["optimal_hidden_layers"].append(inputs[omidx])
        print(optimal_model2)

        obidx = np.argmin(E_gen_s[2*model_amounts:3*model_amounts])+2*model_amounts
        optimal_model3 = M[obidx]
        print(optimal_model3)


        if classify:
            for model_combo in itertools.combinations([optimal_model1, optimal_model2, optimal_model3], 2):
                model1, model2 = model_combo
                model1_name = model1.__class__.__name__
                model2_name = model2.__class__.__name__
                y_pred_model1 = model1.predict(X_test_i)
                y_pred_model2 = model2.predict(X_test_i)
                z_L, z_U, p = compare_regression_models(y_test_i, y_pred_model1, y_pred_model2)
                print(f"Comparing {model1_name} and {model2_name}:")
                print(f"Confidence interval: [{z_L:.4f}, {z_U:.4f}]")
                print(f"p-value: {p:.4f}")

                output_dict[f"{model1_name}_{model2_name}: [zL, zU, p-value]"] = [z_L, z_U, p]

            # Calculate test error on optimal model when tested on D_test_i
            E_test_i_model1 = np.append(E_test_i_model1, sum([a != b for a, b in zip(y_test_i, optimal_model1.predict(X_test_i))]) / len(y_test_i))
            E_test_i_model2 = np.append(E_test_i_model2, sum([a != b for a, b in zip(y_test_i, optimal_model2.predict(X_test_i))]) / len(y_test_i))
            E_test_i_model3 = np.append(E_test_i_model3, sum([a != b for a, b in zip(y_test_i, optimal_model3.predict(X_test_i))]) / len(y_test_i))
        else:
            E_test_i_model1 = np.append(E_test_i_model1, np.square(y_test_i - optimal_model1.predict(X_test_i)).sum() / y_test_i.shape[0])
            E_test_i_model2 = np.append(E_test_i_model2, np.square(y_test_i - optimal_model2.predict(X_test_i)).sum() / y_test_i.shape[0])
            E_test_i_model3 = np.append(E_test_i_model3, np.square(y_test_i - optimal_model3.predict(X_test_i)).sum() / y_test_i.shape[0])


        print(f"E_test_{optimal_model1.__class__.__name__}_{K1_i}:", E_test_i_model1[K1_i-1])
        print(f"E_test_{optimal_model2.__class__.__name__}_{K1_i}:", E_test_i_model2[K1_i-1])
        print(f"E_test_{optimal_model3.__class__.__name__}_{K1_i}:", E_test_i_model3[K1_i-1])

        print()


    print("Outer fold ended.")

    E_gen_model1 = sum((len(D_test_i)/len(y)) * E_test_i_model1[i] for i in range(K1))
    E_gen_model2 = sum((len(D_test_i)/len(y)) * E_test_i_model2[i] for i in range(K1))
    E_gen_model3 = sum((len(D_test_i)/len(y)) * E_test_i_model3[i] for i in range(K1))
    print(f"E_gen_{optimal_model1.__class__.__name__}:", E_gen_model1)
    print(f"E_gen_{optimal_model2.__class__.__name__}:", E_gen_model2)
    print(f"E_gen_{optimal_model3.__class__.__name__}:", E_gen_model3)

    output_dict[f"E_test_{optimal_model1.__class__.__name__}"] = np.round(E_test_i_model1, decimals=3)
    output_dict[f"E_test_{optimal_model2.__class__.__name__}"] = np.round(E_test_i_model2, decimals=3)
    output_dict[f"E_test_{optimal_model3.__class__.__name__}"] = np.round(E_test_i_model3, decimals=3)

    output_dict[f"E_gen_{optimal_model1.__class__.__name__}"] = np.round(E_gen_model1, decimals=3)
    output_dict[f"E_gen_{optimal_model2.__class__.__name__}"] = np.round(E_gen_model2, decimals=3)
    output_dict[f"E_gen_{optimal_model3.__class__.__name__}"] = np.round(E_gen_model3, decimals=3)
    
    return output_dict

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

    def plot_PCA_PCs(self,
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

    def plot_PCA_variance_explained(self,
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

    def plot_PCA_pairs(self,
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

    def plot_PCA_component_coeff(self, pcs: Optional[list] = [], save: bool = True):
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

    def plot_decision_tree(self, criterion = "gini", save: bool = True):
        dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=1.0/self.N)
        dtc = dtc.fit(self.X, self.y)

        # Visualize the graph (you can also inspect the generated image file in an external program)
        # NOTE: depending on your setup you may need to decrease or increase the figsize and DPI setting
        # to get a readable plot. Hint: Try to maximize the figure after it displays.
        fname = f"tree_uci_id_{self.uci_id}_" + criterion + ".png"

        fig = plt.figure(figsize=(4, 4), dpi=100)
        _ = tree.plot_tree(dtc, filled=True, feature_names=self.attributeNames)
        if save:
            plt.savefig(fname, dpi=1000)
        # plt.show()

    def two_step_cross_validation(self, X, y, models: List[BaseEstimator], outer_K: int, inner_K: int) -> None:
        # Set up K-Fold cross-validation
        outer_cv = KFold(n_splits=outer_K, shuffle=True, random_state=42)
        inner_cv = KFold(n_splits=inner_K, shuffle=True, random_state=42)

        # Initialize results dictionary
        results = {model.__class__.__name__: [] for model in models}
        results['Outer fold'] = []

        # Initialize LabelEncoder
        # label_encoder = LabelEncoder()

        # Fit label encoder and return encoded labels
        # y = label_encoder.fit_transform(y.ravel())

        # Start the outer cross-validation loop
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
            X_outer_train, X_outer_test = X[train_idx], X[test_idx]
            y_outer_train, y_outer_test = y[train_idx], y[test_idx]

            # Store best model for each outer fold
            best_model_per_fold = None
            best_model_score = -float('inf')

            for model in models:
                # List to store the inner cross-validation scores for the current model
                inner_scores = []
                
                # Inner cross-validation loop
                for inner_train_idx, inner_val_idx in inner_cv.split(X_outer_train):
                    X_inner_train, X_inner_val = X_outer_train[inner_train_idx], X_outer_train[inner_val_idx]
                    y_inner_train, y_inner_val = y_outer_train[inner_train_idx], y_outer_train[inner_val_idx]

                    # Train the model on the inner train split and evaluate on the inner validation split
                    model.fit(X_inner_train, y_inner_train)
                    y_inner_pred = model.predict(X_inner_val)
                    score = accuracy_score(y_inner_val, y_inner_pred)
                    inner_scores.append(score)
                    #print(inner_scores)

                # Calculate the average inner score for the current model
                avg_inner_score = np.mean(inner_scores)

                # Check if this model is the best one so far and update best_model_per_fold
                if avg_inner_score > best_model_score:
                    best_model_score = avg_inner_score
                    best_model_per_fold = model

            # After the inner loop, retrain the best model on the entire outer train set
            best_model_per_fold.fit(X_outer_train, y_outer_train)
            y_outer_pred = best_model_per_fold.predict(X_outer_test)
            outer_score = accuracy_score(y_outer_test, y_outer_pred)

            # Append the results of the best model to the results dictionary
            model_name = best_model_per_fold.__class__.__name__
            results[model_name].append(outer_score)
            results['Outer fold'].append(fold_idx)

        # After cross-validation, print the results
        for model_name, scores in results.items():
            if model_name != 'Outer fold':
                average_score = np.mean(scores)
                print(f'Average score for {model_name}: {average_score:.4f}')
            else:
                print(f'Outer folds: {scores}')

class Regression:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.X = dataset.X[:, 1:]
        self.y = dataset.X[:, 0]

        self.attributeNames = dataset.attributeNames
        self.attributeNames.remove('Area')

        self.N, self.M = self.X.shape
        self.C = 2

        # Feature transformations
        self.X = self.X - np.ones((self.N, 1)) * self.X.mean(axis=0)
        self.X = self.X * (1 / np.std(self.X, 0))
    
    def two_step(self, max_iter: int = 20000, K: int = 10):
        model_amounts = K
        global inputs
        inputs = []

        astart = -3
        alphas = np.power(10.0, range(astart, astart+model_amounts))
        M = [Ridge(alpha=alpha) for alpha in alphas]
        inputs += alphas.tolist()

        hidden_layer_sizes = [(i,) for i in range(1, model_amounts + 1)]
        M += [MLPRegressor(hidden_layer_sizes=h, max_iter=max_iter) for h in hidden_layer_sizes]
        inputs += hidden_layer_sizes

        M += [DummyRegressor(strategy="mean") for _ in range(model_amounts)]

        print("Alphas:", alphas, sep="\n")
        print("Hidden layer sizes:", hidden_layer_sizes, sep="\n")

        with open("regdata.txt", "w") as f:
            d = two_step_cross_validation(
                X=self.X, y=self.y, 
                M=M,
                K1=K, K2=K,
                model_amounts = model_amounts)
            
            for key, value in d.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\n\n")

class Classification:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.X = dataset.X
        self.y = dataset.y

        self.y = (self.y.ravel() == 'Osmancik').astype(int)

        self.attributeNames = dataset.attributeNames

        self.N = dataset.N
        self.M = dataset.M
        self.C = 2

        # Feature transformations
        self.X = self.X - np.ones((self.N, 1)) * self.X.mean(axis=0)
        self.X = self.X * (1 / np.std(self.X, 0))
    
    def two_step(self, max_iter: int = 20000, K: int = 10):
        model_amounts = K
        global inputs
        inputs = []

        astart = -5
        C = np.power(10.0, range(astart, astart+model_amounts))
        M = [LogisticRegression(C=c, penalty='l2') for c in C]
        inputs += C.tolist()

        hidden_layer_sizes = [(i,) for i in range(1, model_amounts + 1)]
        M += [MLPClassifier(hidden_layer_sizes=h, max_iter=max_iter) for h in hidden_layer_sizes]
        inputs += hidden_layer_sizes

        M += [DummyClassifier(strategy="most_frequent") for _ in range(model_amounts)]

        print("C:", C)
        print("Hidden layer sizes:", hidden_layer_sizes, sep="\n")

        with open("classifydata.txt", "w") as f:
            d = two_step_cross_validation(
                X=self.X, y=self.y, 
                M=M,
                classify=True,
                K1=K, K2=K,
                model_amounts = model_amounts)
            
            for key, value in d.items():
                f.write(f"{key}: {value}\n")

            f.write("\n\n")

if __name__ == "__main__":
    dataset = Dataset(uci_id = 545)
    # Features: Area, Perimeter, Major_Axis_Length, Minor_Axis_Length, Eccentricity, Convex_Area, Extent
    # Targets: Class (Cammeo, Osmancik)

    # logistic_model = LogisticRegression(max_iter=1000)  # Add any specific hyperparameters you need

    # # Define your models
    # # models = [MLPClassifier(...), LogisticRegression(...), DummyClassifier(...)]

    # # Add the models to a list
    # models = [logistic_model]  # Replace ... with other models instances if you have any

    # Perform the two-step cross-validation
    # dataset.two_step_cross_validation(models=models, K1=10, K2=10)

    #print(dataset.y)

    # data = Regression(dataset)
    data = Classification(dataset)
    data.two_step(max_iter=10, K=10)



