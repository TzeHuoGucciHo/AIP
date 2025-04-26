import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import torch
import torch.nn as nn
import torch.optim as optim


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from torch.fx.experimental.proxy_tensor import extract_val
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import TensorDataset, DataLoader

import warnings

warnings.filterwarnings('ignore')

dataframe = pd.read_csv(r"C:\Users\Tze Huo Gucci Ho\Desktop\Git Projects\AIP\Exc_06\Data\train.csv")

    # Data Overview

#print(dataframe.head())

#print(dataframe.shape)

#print(dataframe.info())

#print(dataframe.describe())

    # Data Cleaning

#dataframe.isnull().sum().plot.bar()
#plt.show()

col = dataframe.loc[:,'RoomService':'VRDeck'].columns
#print(dataframe.groupby('VIP')[col].mean())

#print(dataframe.groupby('CryoSleep')[col].mean())

temp = dataframe['CryoSleep'] == True
dataframe.loc[temp, col] = 0.0

for c in col:
    for val in [True, False]:
        temp = dataframe['VIP'] == val

        k = dataframe.loc[temp, c].astype(float).mean()
        dataframe.loc[temp, c] = dataframe.loc[temp, c].fillna(k)

#sb.countplot(data=dataframe, x='VIP',
#             hue='HomePlanet')
#plt.show()

col = 'HomePlanet'
temp = dataframe['VIP'] == False
dataframe.loc[temp, col] = dataframe.loc[temp, col].fillna('Earth')

temp = dataframe['VIP'] == True
dataframe.loc[temp, col] = dataframe.loc[temp, col].fillna('Europa')

# sb.boxplot(dataframe['Age'],orient='h')
#plt.show()

temp = dataframe[dataframe['Age'] < 61]['Age'].mean()
dataframe['Age'] = dataframe['Age'].fillna(temp)

#sb.countplot(data=dataframe,
#             x='Transported',
#             hue='CryoSleep')
#plt.show()

#dataframe.isnull().sum().plot.bar()
#plt.show()

for col in dataframe.columns:
    if dataframe[col].isnull().sum() == 0:
        continue

    if dataframe[col].dtype == object or dataframe[col].dtype == bool:
        dataframe[col] = dataframe[col].fillna(dataframe[col].mode()[0])

    else:
        dataframe[col] = dataframe[col].fillna(dataframe[col].mean())

#print("Number of Null Values in Dataset: ", dataframe.isnull().sum().sum())

    # Feature Engineering

new = dataframe["PassengerId"].str.split("_", n=1, expand=True)
dataframe["RoomNo"] = new[0].astype(int)
dataframe["PassengerNo"] = new[1].astype(int)

dataframe.drop(['PassengerId', 'Name'],
        axis=1, inplace=True)

data = dataframe['RoomNo']
for i in range(dataframe.shape[0]):
      temp = data == data[i]
      dataframe['PassengerNo'][i] = (temp).sum()

dataframe.drop(['RoomNo'], axis=1,
        inplace=True)

#sb.countplot(data=dataframe,
#             x = 'PassengerNo',
#             hue='VIP')
#plt.show()

new = dataframe["Cabin"].str.split("/", n=2, expand=True)
data["F1"] = new[0]
dataframe["F2"] = new[1].astype(int)
dataframe["F3"] = new[2]

dataframe.drop(['Cabin'], axis=1,
        inplace=True)

dataframe['LeasureBill'] = dataframe['RoomService'] + dataframe['FoodCourt']\
 + dataframe['ShoppingMall'] + dataframe['Spa'] + dataframe['VRDeck']

    # Exploratory Data Analysis

x = dataframe['Transported'].value_counts()
#plt.pie(x.values,
#        labels=x.index,
#        autopct='%1.1f%%')
#plt.show()

dataframe.groupby('VIP').mean(numeric_only=True)['LeasureBill'].plot.bar()
#plt.show()

for col in dataframe.columns:

    if dataframe[col].dtype == object:
        le = LabelEncoder()
        dataframe[col] = le.fit_transform(dataframe[col])

    if dataframe[col].dtype == 'bool':
        dataframe[col] = dataframe[col].astype(int)

#print(dataframe.head())

#plt.figure(figsize=(10,10))
#sb.heatmap(dataframe.corr()>0.8,
#           annot=True,
#           cbar=False)
#plt.show()

    # Model Training

features = dataframe.drop(['Transported'], axis=1)
target = dataframe.Transported

X_train, X_val,\
    Y_train, Y_val = train_test_split(features, target,
                                      test_size=0.1,
                                      random_state=22)

#print(X_train.shape, X_val.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

    # Training using scikit-learn

from sklearn.metrics import roc_auc_score as ras

models = [LogisticRegression(), XGBClassifier(),
          SVC(kernel='rbf', probability=True)]

for i in range(len(models)):
    models[i].fit(X_train, Y_train)

#    print(f'{models[i]} : ')

    train_preds = models[i].predict_proba(X_train)[:, 1]
#    print('Training Accuracy : ', ras(Y_train, train_preds))

    val_preds = models[i].predict_proba(X_val)[:, 1]
#    print('Validation Accuracy : ', ras(Y_val, val_preds))
#    print()

    # Model Evaluation

y_pred = models[1].predict(X_val)
cm = metrics.confusion_matrix(Y_val, y_pred)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
#disp.plot()
#plt.show()

#print(metrics.classification_report
#      (Y_val, models[1].predict(X_val)))

    # Transform data ("Transported") to PyTorch tensors

data_numpy = dataframe.to_numpy(dtype=float)

data_tensor = torch.tensor(data_numpy, dtype=torch.float32)

#print(data_tensor.shape)
    # Outputs: torch.Size([8693, 15])

x = dataframe.drop(columns=["Transported"]).to_numpy(dtype=float)
y = dataframe["Transported"].to_numpy(dtype=float)

x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# print(x_tensor.shape, y_tensor.shape)

dataset = TensorDataset(x_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Create a simple neural network with 2 layers

# Define the neural network class
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(SimpleNN, self).__init__()
        # Define layers: 1st layer - fully connected layer (input_size -> hidden_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Second layer - fully connected layer (hidden_size -> output size, which is 1 for binary classification)
        self.fc2 = nn.Linear(hidden_size, 1)
        # Activation function (ReLU for the first layer)
        self.relu = nn.ReLU()
        # Sigmoid activation for binary classification output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Apply first layer and ReLU activation
        x = self.fc2(x)             # Apply second layer
        return self.sigmoid(x)      # Apply Sigmoid activation for binary classification

# Model initialization
input_size = X_train.shape[1]  # Number of features
model = SimpleNN(input_size)

# Loss function (Binary Cross-Entropy loss)
criterion = nn.BCEWithLogitsLoss()

# Optimizer (Stochastic Gradient Descent)
optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-8)

# Training loop
epochs = 1000  # Number of epochs
train_losses = []
train_accuracies = []

for epoch in range(epochs):
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Zero the gradients from the previous step

    # Forward pass
    outputs = model(torch.tensor(X_train, dtype=torch.float32))
    loss = criterion(outputs.squeeze(), torch.tensor(Y_train, dtype=torch.float32))  # Compute the loss

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Record training loss and accuracy
    train_losses.append(loss.item())

    # Calculate accuracy
    predicted = (outputs.squeeze() > 0.5).float()
    accuracy = (predicted == torch.tensor(Y_train, dtype=torch.float32)).float().mean()
    train_accuracies.append(accuracy.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}')

# Plotting the loss curve
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Plotting the loss curve
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Plotting the accuracy curve
plt.plot(train_accuracies)
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

# Evaluate the model on the validation set
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    # Forward pass on validation set
    val_outputs = model(torch.tensor(X_val, dtype=torch.float32))
    val_preds = (val_outputs.squeeze() > 0.5).float()

# Compute the classification metrics
accuracy = accuracy_score(Y_val, val_preds.numpy())
precision = precision_score(Y_val, val_preds.numpy())
recall = recall_score(Y_val, val_preds.numpy())
f1 = f1_score(Y_val, val_preds.numpy())

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')
