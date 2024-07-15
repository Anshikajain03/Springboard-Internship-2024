import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load CSV data
df = pd.read_csv('pushup_keypoints.csv')

# Step 2: Handle missing values (if any)
# Check for missing values
missing_values = df.isnull().sum()
print("Missing values:\n", missing_values)

# Optionally, impute missing values
# df.fillna(df.mean(), inplace=True)  # Example: impute with mean

# Step 3: Feature Engineering (if needed)
# Example: Calculate distance between wrist and shoulder
# df['wrist_shoulder_distance'] = ((df['wrist_left_x'] - df['shoulder_left_x']) ** 2 + (df['wrist_left_y'] - df['shoulder_left_y']) ** 2) ** 0.5

# Step 4: Normalize or Standardize Data (if needed)
# Example: Standardize numeric columns
scaler = StandardScaler()
numeric_columns = df.columns.drop('exercise')  # Assuming 'exercise' is the target variable
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Step 5: Split data into training and testing sets
X = df.drop('exercise', axis=1)
y = df['exercise']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now X_train, X_test, y_train, y_test are ready for training and testing your model
