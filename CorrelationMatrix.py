import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

# Load your dataset
df = pd.read_csv(r"C:\Users\User\Desktop\cleaned_training_dataset_rounded.csv")

# Calculate correlation matrix
correlation_matrix = df.corr()
print(correlation_matrix)

# Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Matrix of Input Features", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Show plot
plt.show()
