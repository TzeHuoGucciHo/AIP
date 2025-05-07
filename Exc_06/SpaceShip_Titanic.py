import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore")

### Exercise 6.a

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

## Exercise 6.a.1
train_path = r"C:\Users\Tze Huo Gucci Ho\Desktop\Git Projects\AIP\Exc_06\Data\train.csv"
test_path = r"C:\Users\Tze Huo Gucci Ho\Desktop\Git Projects\AIP\Exc_06\Data\test.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# Data Preprocessing & Feature Engineering

def preprocess(df, is_train=True):
    df = df.copy()

    # Fill CryoSleep-related expenses
    col = df.loc[:, 'RoomService':'VRDeck'].columns
    df.loc[df['CryoSleep'] == True, col] = 0.0

    for c in col:
        for val in [True, False]:
            temp = df['VIP'] == val
            k = df.loc[temp, c].astype(float).mean()
            df.loc[temp, c] = df.loc[temp, c].fillna(k)

    # Fill HomePlanet based on VIP status
    col = 'HomePlanet'
    df.loc[df['VIP'] == False, col] = df.loc[df['VIP'] == False, col].fillna('Earth')
    df.loc[df['VIP'] == True, col] = df.loc[df['VIP'] == True, col].fillna('Europa')

    # Fill Age
    df['Age'] = df['Age'].fillna(df[df['Age'] < 61]['Age'].mean())

    # Fill remaining missing values
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if df[col].dtype == object or df[col].dtype == bool:
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())

    # Feature Engineering
    new = df["PassengerId"].str.split("_", n=1, expand=True)
    df["RoomNo"] = new[0].astype(int)
    df["PassengerNo"] = new[1].astype(int)
    df.drop(['PassengerId', 'Name'], axis=1, inplace=True)

    # Count passengers with same room number
    data = df["RoomNo"]
    for i in range(df.shape[0]):
        temp = data == data.iloc[i]
        df['PassengerNo'].iloc[i] = temp.sum()

    df.drop(['RoomNo'], axis=1, inplace=True)

    # Extract Cabin features
    new = df["Cabin"].str.split("/", n=2, expand=True)
    df["F1"] = new[0]
    df["F2"] = new[1].astype(int)
    df["F3"] = new[2]
    df.drop(['Cabin'], axis=1, inplace=True)

    # Total expenses feature
    df['LeasureBill'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']

    # Encode categorical variables
    for col in df.columns:
        if df[col].dtype == object:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)

    return df

df_train = preprocess(df_train, is_train=True)
df_test = preprocess(df_test, is_train=False)

### Exercise 6.b

## Exercise 6.b.1

# Convert data to PyTorch tensors
# Split training data into features and labels
X = df_train.drop('Transported', axis=1).values
y = df_train['Transported'].values

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # reshape for binary classification

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)  # reshape for binary classification

## Exercise 6.b.2

# Define the neural network class
class ImprovedNet(nn.Module):
    def __init__(self, input_dim):
        super(ImprovedNet, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.3)

        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.3)

        self.output = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.dropout3(self.relu3(self.fc3(x)))
        x = self.sigmoid(self.output(x))
        return x


# Initialize the model

input_dim = X_train_tensor.shape[1]  # Number of features
hidden_dim1 = 32                # You can start simple; try 32 and tune later
hidden_dim2 = 16                # You can start simple; try 16 and tune later

## Exercise 6.b.3, 6.b.4

# Define loss function and optimizer
model = ImprovedNet(input_dim=X_train_tensor.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)

# Training loop
num_epochs = 1000
accuracy_list = []

# Initialize a list to store the loss values during training
loss_list = []
val_loss_list = []
accuracy_list_train = []
accuracy_list_val = []

for epoch in range(num_epochs):
    # Training predictions
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Validation predictions
    outputs_val = model(X_val_tensor)
    val_loss = criterion(outputs_val, y_val_tensor)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Accuracy
    predicted_train = (outputs.detach().numpy() > 0.5).astype(int)
    accuracy_train = metrics.accuracy_score(y_train_tensor.numpy(), predicted_train)

    predicted_val = (outputs_val.detach().numpy() > 0.5).astype(int)
    accuracy_val = metrics.accuracy_score(y_val_tensor.numpy(), predicted_val)

    # Store metrics
    loss_list.append(loss.item())
    val_loss_list.append(val_loss.item())
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

## Exercise 6.b.5

# Plot training loss
plt.figure(figsize=(10, 4))

# --- Loss plot ---
plt.subplot(1, 2, 1)
plt.plot(range(num_epochs), loss_list, label='Training Loss', color='red')
plt.plot(range(num_epochs), val_loss_list, label='Validation Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()
plt.grid(True)

# --- Accuracy plot ---
plt.subplot(1, 2, 2)
plt.plot(range(num_epochs), accuracy_list_train, label='Training Accuracy', color='blue')
plt.plot(range(num_epochs), accuracy_list_val, label='Validation Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


## Exercise 6.b.6

# Make predictions on the validation set
outputs_val = model(X_val_tensor)
predicted_val = (outputs_val.detach().numpy() > 0.5).astype(int)

# Calculate precision, recall, and F1-score on the validation set
precision = precision_score(y_val_tensor.numpy(), predicted_val)
recall = recall_score(y_val_tensor.numpy(), predicted_val)
f1 = f1_score(y_val_tensor.numpy(), predicted_val)

print(f"Validation Precision: {precision:.4f}")
print(f"Validation Recall: {recall:.4f}")
print(f"Validation F1-score: {f1:.4f}")

# Optionally, you can also calculate these metrics on the training data (for monitoring during training)
outputs_train = model(X_train_tensor)
predicted_train = (outputs_train.detach().numpy() > 0.5).astype(int)

precision_train = precision_score(y_train_tensor.numpy(), predicted_train)
recall_train = recall_score(y_train_tensor.numpy(), predicted_train)
f1_train = f1_score(y_train_tensor.numpy(), predicted_train)

print(f"Training Precision: {precision_train:.4f}")
print(f"Training Recall: {recall_train:.4f}")
print(f"Training F1-score: {f1_train:.4f}")

# Plot comparison of training and validation metrics
metrics_names = ['Precision', 'Recall', 'F1-score']
training_scores = [precision_train, recall_train, f1_train]
validation_scores = [precision, recall, f1]

x = np.arange(len(metrics_names))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, training_scores, width, label='Training')
bars2 = ax.bar(x + width/2, validation_scores, width, label='Validation')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Model Performance Metrics')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()

# Annotate the bars with their values
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # offset text above bar
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.ylim(0, 1)  # set y-axis from 0 to 1 for clarity
plt.tight_layout()
plt.show()
