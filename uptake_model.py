import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns


# Load the data
file_path = 'uptake_data.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Combine specified datasets
train_data_combined = pd.concat([
    data[data['filename'].str.startswith('DB13-')],
    data[data['filename'].str.startswith('DB12-')],
    data[data['filename'].str.startswith('DB0-')]
])

test_data_db13 = pd.concat([
    data[data['filename'].str.startswith('DB1-')],
    data[data['filename'].str.startswith('DB5-')],
    data[data['filename'].str.startswith('DB14-')],
    data[data['filename'].str.startswith('DB8-')],
    data[data['filename'].str.startswith('DB7-')]
])

# Prepare training and testing sets
X_train_combined = train_data_combined.drop(columns=['filename', ' Average loading absolute [mol/kg framework] Component 0']).select_dtypes(include=[np.number])
y_train_combined = train_data_combined[' Average loading absolute [mol/kg framework] Component 0']

# Preparing the DB13 test set
X_test_db13 = test_data_db13.drop(columns=['filename', ' Average loading absolute [mol/kg framework] Component 0']).select_dtypes(include=[np.number])
y_test_db13 = test_data_db13[' Average loading absolute [mol/kg framework] Component 0']

# Function to evaluate the model
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    return mse, r2, mae, y_pred

# Final model with best hyperparameters
cb_model = CatBoostRegressor(
    iterations=300,
    depth=5,
    learning_rate=0.1,
    l2_leaf_reg=1,
    random_state=42,
    verbose=False  # Disable logging for clarity
)

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X_train_combined, y_train_combined, test_size=0.2, random_state=133)

# Train the model
cb_model.fit(X_train, y_train)

# Evaluate performance on training, validation, and DB13 sets
mse_train, r2_train, mae_train, y_train_pred = evaluate_model(cb_model, X_train, y_train)
mse_val, r2_val, mae_val, y_val_pred = evaluate_model(cb_model, X_val, y_val)
mse_test_db13, r2_test_db13, mae_test_db13, y_test_db13_pred = evaluate_model(cb_model, X_test_db13, y_test_db13)

# Output the evaluation results
print(f'Training R²: {r2_train}, MAE: {mae_train}')
print(f'Test R²: {r2_val}, MAE: {mae_val}')
print(f'Cross-diversity R²: {r2_test_db13}, MAE: {mae_test_db13}')

# Create DataFrames for each set
df_train = pd.DataFrame({
    'TRUE': y_train.values,
    'Predicted': y_train_pred,
    'Set': 'Train'
})

df_val = pd.DataFrame({
    'TRUE': y_val.values,
    'Predicted': y_val_pred,
    'Set': 'Test'
})

df_test_db13 = pd.DataFrame({
    'TRUE': y_test_db13.values,
    'Predicted': y_test_db13_pred,
    'Set': 'Cross-diversity'
})

# Combine the DataFrames
df_all = pd.concat([df_train, df_val, df_test_db13], ignore_index=True)

# Proceed with the plotting code, using df_all

# Convert 'TRUE' and 'Predicted' columns to numeric values to ensure proper data handling
df_all['TRUE'] = pd.to_numeric(df_all['TRUE'], errors='coerce')
df_all['Predicted'] = pd.to_numeric(df_all['Predicted'], errors='coerce')

# Remove any rows with NaNs after the conversion
clean_data = df_all.dropna(subset=['TRUE', 'Predicted'])

# Get unique categories in 'Set' column
categories = clean_data['Set'].unique()

# Define colors for each dataset
palette = {
    'Train': '#D3D3D3',           # Very light grey
    'Test': '#4A9DE9',            # Deeper cool blue
    'Cross-diversity': '#E97A6E'  # Deeper warm red/pink
}

# Check that all categories are in the palette
missing_categories = set(categories) - set(palette.keys())
if missing_categories:
    raise ValueError(f"Colors not specified for categories: {missing_categories}")

# Create the plot with two KDE plots and a scatter plot
fig, ax = plt.subplots(
    2, 2, figsize=(7, 7), dpi=600,
    gridspec_kw={'height_ratios': [1, 8], 'width_ratios': [8, 1]}
)

# Assign axes for easier reference
scatter_ax = ax[1, 0]
x_kde_ax = ax[0, 0]
y_kde_ax = ax[1, 1]
empty_ax = ax[0, 1]

# Scatter plot with transparency for True vs Predicted
sns.scatterplot(
    data=clean_data,
    x='TRUE',
    y='Predicted',
    hue='Set',
    ax=scatter_ax,
    palette=palette,
    alpha=0.3,
    edgecolor=None
)

# Add regression lines for each category without adding to legend
for category in categories:
    subset = clean_data[clean_data['Set'] == category]
    sns.regplot(
        data=subset,
        x='TRUE',
        y='Predicted',
        ax=scatter_ax,
        scatter=False,
        color=palette[category],
        line_kws={'linestyle': '--'},
        label='_nolegend_'  # Prevents adding to legend
    )

# Adjust legend to only show data points, not trendlines
scatter_ax.legend(title=None, frameon=False, prop={'family': 'cmss10', 'size': 14})

# KDE for the True values on the top with shaded area
sns.kdeplot(
    data=clean_data,
    x='TRUE',
    hue='Set',
    ax=x_kde_ax,
    palette=palette,
    alpha=0.5,
    fill=True,
    legend=False
)

# KDE for the Predicted values on the right with shaded area
sns.kdeplot(
    data=clean_data,
    y='Predicted',
    hue='Set',
    ax=y_kde_ax,
    palette=palette,
    alpha=0.5,
    fill=True,
    legend=False
)

# Turn off the top right empty plot
empty_ax.axis('off')

# Remove ticks and labels for the KDE plots
x_kde_ax.set_xticklabels([])
x_kde_ax.set_xlabel('')
x_kde_ax.set_yticks([])
x_kde_ax.set_ylabel('')

y_kde_ax.set_yticklabels([])
y_kde_ax.set_ylabel('')
y_kde_ax.set_xticks([])
y_kde_ax.set_xlabel('')

# Remove spines for KDE plots
sns.despine(ax=x_kde_ax, left=True, bottom=True)
sns.despine(ax=y_kde_ax, left=True, bottom=True)

# Adjust main scatter plot
scatter_ax.set_xlabel('True Values', fontsize=12)
scatter_ax.set_ylabel('Predicted Values', fontsize=12)

# Set specific x and y ticks
scatter_ax.set_xticks([5, 10, 15, 20])
scatter_ax.set_yticks([5, 10, 15, 20])

# Remove top and right spines from the main scatter plot
sns.despine(ax=scatter_ax)

# Adjust layout for better spacing
plt.tight_layout()

plt.savefig('true_vs_predicted_plot_uptake.png')
# Display the plot
plt.show()
