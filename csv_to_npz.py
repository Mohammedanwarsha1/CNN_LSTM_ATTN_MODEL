import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import KFold

# Load the CSV data
csv_file_path = r'E:\yoga\Yoga-Pose-Classification-and-Skeletonization\encodingHumanActivity\train.csv'
data = pd.read_csv(csv_file_path)

# Extract features and labels
X = data.iloc[:, 1:].values  # All columns except the first one
y = data.iloc[:, 0].values  # The first column

# Convert labels to one-hot encoding
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y)
onehot_encoder = OneHotEncoder(sparse_output=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
y_one_hot = onehot_encoder.fit_transform(integer_encoded)

# Create folds for cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=0)
folds = [(train_idx, test_idx) for train_idx, test_idx in kf.split(X)]

# Convert folds to a numpy array with dtype=object
folds_array = np.array(folds, dtype=object)

# Save the data in .npz format
np.savez('squat_exercise.npz', X=X, y=y_one_hot, folds=folds_array)