import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"C:\Users\User\Desktop\cleaned_training_dataset.csv")

# Correlation check
correlation = df.corr(numeric_only=True)['Class'].sort_values(ascending=False)
print("Correlation of features with target:\n", correlation)

# Check duplicates
print("Duplicate rows:", df.duplicated().sum())

# Fix PyCharm Matplotlib issue
plt.switch_backend('TkAgg')

# Boxplot
sns.boxplot(data=df, x='Class', y='Seismicity_PGA')
plt.title("Boxplot: Seismicity_PGA vs Class")
plt.show()
