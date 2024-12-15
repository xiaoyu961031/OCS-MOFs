import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def compute_js_divergence_2d(data1, data2, gridsize=100):
    """
    Computes the Jensen-Shannon Divergence between two 2D distributions
    (data1 and data2) by first estimating their kernel density.
    """
    kde1 = gaussian_kde(data1.T)
    kde2 = gaussian_kde(data2.T)
    
    # Create a grid covering the range of both datasets
    x_min = min(data1[:,0].min(), data2[:,0].min())
    x_max = max(data1[:,0].max(), data2[:,0].max())
    y_min = min(data1[:,1].min(), data2[:,1].min())
    y_max = max(data1[:,1].max(), data2[:,1].max())

    x_grid = np.linspace(x_min, x_max, gridsize)
    y_grid = np.linspace(y_min, y_max, gridsize)
    X, Y = np.meshgrid(x_grid, y_grid)
    coords = np.vstack([X.ravel(), Y.ravel()])

    # Evaluate the KDE on the grid
    pdf1 = kde1(coords).reshape(gridsize, gridsize)
    pdf2 = kde2(coords).reshape(gridsize, gridsize)

    # Flatten the PDFs
    pdf1_flat = pdf1.ravel()
    pdf2_flat = pdf2.ravel()

    # Normalize them so they sum to 1
    pdf1_flat /= np.sum(pdf1_flat)
    pdf2_flat /= np.sum(pdf2_flat)

    # Compute Jensen-Shannon divergence
    js_div = jensenshannon(pdf1_flat, pdf2_flat)
    return js_div

# -------------------------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------------------------
data = pd.read_csv('TSNE.csv')  # Replace with your actual CSV file name

# -------------------------------------------------------------------------------
# 2. CREATE TRAIN/TEST SPLITS
# -------------------------------------------------------------------------------
train_data_combined = pd.concat([
    data[data['filename'].str.startswith('DB12-')],
    data[data['filename'].str.startswith('DB13-')],
    data[data['filename'].str.startswith('DB0-')]
])

test_data_db13 = pd.concat([
    data[data['filename'].str.startswith('DB1-')],
    data[data['filename'].str.startswith('DB5-')],
    data[data['filename'].str.startswith('DB14-')],
    data[data['filename'].str.startswith('DB8-')],
    data[data['filename'].str.startswith('DB7-')]
])

# Add labels
train_data_combined['data_type'] = 'train'
test_data_db13['data_type'] = 'test'

combined_data = pd.concat([train_data_combined, test_data_db13]).reset_index(drop=True)

# Convert 'data_type' to numeric labels: 0=Train, 1=Test
combined_data['label'] = combined_data['data_type'].map({'train':0, 'test':1})

# Extract the t-SNE coordinates into numpy arrays
train_points = train_data_combined[['tsne_dim1','tsne_dim2']].values
test_points  = test_data_db13[['tsne_dim1','tsne_dim2']].values

# For combined metrics
X_combined = combined_data[['tsne_dim1','tsne_dim2']].values
y_combined = combined_data['label'].values

# -------------------------------------------------------------------------------
# 3. QUANTITATIVE MEASURES
# -------------------------------------------------------------------------------

print("=== 1) Distance Between Group Centroids ===")
train_centroid = np.mean(train_points, axis=0)
test_centroid  = np.mean(test_points, axis=0)
centroid_distance = np.linalg.norm(train_centroid - test_centroid)
print(f"Distance between centroids: {centroid_distance:.4f}\n")


print("=== 2) Average (Pairwise) Inter-Group Distances ===")
dist_matrix = cdist(train_points, test_points, metric='euclidean')
avg_distance = dist_matrix.mean()
print(f"Average pairwise distance (train vs. test): {avg_distance:.4f}\n")


print("=== 3) Nearest Neighbor Distances ===")
# Nearest neighbor from train to test
min_distances_train = np.min(dist_matrix, axis=1)
# Nearest neighbor from test to train
min_distances_test = np.min(dist_matrix, axis=0)

avg_min_distance_train = np.mean(min_distances_train)
avg_min_distance_test  = np.mean(min_distances_test)

print(f"Average min distance (train->test): {avg_min_distance_train:.4f}")
print(f"Average min distance (test->train): {avg_min_distance_test:.4f}\n")


print("=== 4) Clustering Separation (Silhouette Score) ===")
sil_score = silhouette_score(X_combined, y_combined, metric='euclidean')
print(f"Silhouette Score (range -1 to 1): {sil_score:.4f}\n")


print("=== 5) Distribution-Based Separation (Jensen-Shannon Divergence) ===")
js_div_value = compute_js_divergence_2d(train_points, test_points, gridsize=100)
print(f"Jensen-Shannon Divergence: {js_div_value:.4f}\n")


print("=== 6) Classification-Based Separation (Logistic Regression) ===")
# We'll split the combined data again into some small train/val partition 
# just for measuring classification accuracy in the t-SNE space
X_train, X_val, y_train, y_val = train_test_split(
    X_combined, y_combined, test_size=0.3, random_state=42
)
clf = LogisticRegression()
clf.fit(X_train, y_train)
preds = clf.predict(X_val)
acc = accuracy_score(y_val, preds)
print(f"Logistic Regression accuracy in 2D space: {acc:.4f}\n")

# -------------------------------------------------------------------------------
# 4. OPTIONAL: VISUALIZE
# -------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5,4), dpi=120)
color_map = {0:'#4A9DE9', 1:'#E97A6E'}  # train=blue, test=red
scatter_colors = [color_map[label] for label in y_combined]

sc = ax.scatter(
    X_combined[:,0], 
    X_combined[:,1], 
    c=scatter_colors, 
    alpha=0.5,
    edgecolor='k',
    linewidth=0.2
)

ax.set_title("t-SNE plot with train (blue) vs test (red)")
ax.set_xlabel("tsne_dim1")
ax.set_ylabel("tsne_dim2")
plt.show()
