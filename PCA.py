import matplotlib.pyplot as plt
from Import import *
from scipy.linalg import svd


Y = X - np.ones((N, 1)) * X.mean(axis=0)
Y = Y * (1 / np.std(Y, 0))

U, S, Vh = svd(Y, full_matrices=False)
rho = (S * S) / (S * S).sum()

V = Vh.T
# Project the centered data onto principal component space
Z = Y @ V
# Indices of the principal components to be plotted
# i,j are principal components i,j:
i = 2
j = 3

f = plt.figure()
plt.title("Rice: PCA")
# Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y == c
    plt.plot(Z[class_mask, i], Z[class_mask, j], "o", alpha=0.5)
plt.legend(classNames)
plt.xlabel("PC{0}".format(i + 1))
plt.ylabel("PC{0}".format(j + 1))

# Output result to screen


threshold = 0.9
# Plot variance explained
plt.figure()
plt.plot(range(1, len(rho) + 1), rho, "x-")
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
plt.plot([1, len(rho)], [threshold, threshold], "k--")
plt.title("Variance explained by principal components")
plt.xlabel("Principal component")
plt.ylabel("Variance explained")
plt.legend(["Individual", "Cumulative", "Threshold"])
plt.grid()


plt.show()



