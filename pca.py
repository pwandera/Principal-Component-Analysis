import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
np.random.seed(1)
sns.set_style('whitegrid')
warnings.filterwarnings('ignore')

countries = pd.read_csv('countries/Country-data.csv', index_col = 'country')
countries

n, p = countries.shape
print(f'Observations (Countries): {n}')
print(f'Variables: {p}')
print()

descriptions = pd.read_csv('countries/data-dictionary.csv')

print('Variable Descriptions')
for i in range(1, descriptions.shape[0]):
    print(f'X{i} {descriptions.iloc[i, 0]} : {descriptions.iloc[i, 1]}')

X = countries.values

# Sample Mean Vector
mu = (1/n) * X.T @ np.ones((n,1))
print('Mean Vector: ')
print(pd.DataFrame(mu, index = countries.columns))

# Sample Covariance Matrix (S)
Q = np.ones((n, 1)) @ mu.T
S = (1/(n-1)) * (X - Q).T @  (X - Q)

print('\nCovariance Matrix:')
pd.DataFrame(S, index = countries.columns, columns = countries.columns)

# Sample Correlation Matrix (R)
V = np.sqrt(np.diag(np.diag(S)))
V_inv = np.linalg.inv(V)
R = V_inv @ S @ V_inv

# Standardized Data Matrix
Z = (X - Q) @ V_inv

corr_eigen_values, corr_eigen_vectors = np.linalg.eig(R)

# Sorting Eigen-Decomposition in order of greatest variance (eigenvalue)
sorted_indices = np.argsort(corr_eigen_values)[::-1]
corr_eigen_values = corr_eigen_values[sorted_indices]
corr_eigen_vectors = corr_eigen_vectors[:, sorted_indices]

total_variance = sum(corr_eigen_values)
total_variance

for i in range(p):
    print(f'PC{i+1} : {100 * corr_eigen_values[i] / total_variance:.5f}%')

explained_variance = 100 * corr_eigen_values / total_variance
print(f'\nPC1, PC2 & PC3 % of Explained Variance: {sum(explained_variance[0:3]):.5f}%')
print(f'PC1, PC2 & PC3 % of lost variance: {100 - sum(explained_variance[0:3]):.5f}%\n')

for i in range(3):
    print(f'PC{i+1} = ', end = '')
    for j in range(p):
        print(f'{corr_eigen_vectors[j][i]:.4f} * X{j+1} ', end = '')
        if (j != p - 1):
            print('+ ', end = '')
    print()

Y = Z @ corr_eigen_vectors[:, 0:3]

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# ax.set_title('Data Projected on Principal Components')
# ax.scatter(xs = Y[:, 0], ys = Y[:, 1], zs = Y[:, 2], c = 'r')
# ax.set_xlabel(f'PC1: {explained_variance[0]:.1f}%')
# ax.set_ylabel(f'PC2: {explained_variance[1]:.1f}%')
# ax.set_zlabel(f'PC3: {explained_variance[2]:.1f}%')

# for i in range(n):
#     text = countries.index[i]
#     ax.text(Y[i, 0], Y[i, 1], Y[i, 2], text, size = 7.5)

# plt.show()

pca_countries = pd.DataFrame(Y, columns = ['PC1', 'PC2', 'PC3'])
pca_countries['COUNTRY'] = countries.index
pca_countries = pca_countries[['COUNTRY', 'PC1', 'PC2', 'PC3']]
pca_countries

sample = pca_countries.sample(n = 20)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.set_title('Data Projected on Principal Components')
ax.scatter(xs = sample.iloc[:, 1], ys = sample.iloc[:, 2], zs = sample.iloc[:, 3], c = 'r')
ax.set_xlabel(f'PC1: {explained_variance[0]:.1f}%')
ax.set_ylabel(f'PC2: {explained_variance[1]:.1f}%')
ax.set_zlabel(f'PC3: {explained_variance[2]:.1f}%')

for i in range(20):
    text = sample.iloc[i, 0]
    ax.text(sample.iloc[i, 1], sample.iloc[i, 2], sample.iloc[i, 3], text, size = 10)

plt.show()
