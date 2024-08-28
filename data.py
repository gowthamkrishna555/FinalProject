import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
from streamlit_folium import folium_static
from sklearn.metrics import accuracy_score

# Load labeled dataset
data = pd.read_csv('Karnataka_Watershed_Labeled.csv')

# Extract features and target variable
features = data[['COAST', 'DIST_MAIN', 'DIST_SINK', 'ENDO', 'HYBAS_ID', 'MAIN_BAS', 'NEXT_DOWN', 'NEXT_SINK', 'ORDER_', 'PFAF_ID', 'SORT', 'SUB_AREA', 'UP_AREA']]
target = data['wetland_label']

# Standardize features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Convert data to PyTorch tensors
X = torch.FloatTensor(features)
y = torch.FloatTensor(target.values)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Initialize the neural network
input_size = X_train.shape[1]
model = SimpleNN(input_size)

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 50
for epoch in range(epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train.view(-1, 1))

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Make predictions on the test set
test_predictions = model(X_test).detach().numpy().flatten()

# Convert probabilities to binary predictions (0 or 1)
binary_predictions = (test_predictions > 0.5).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test.numpy().astype(int), binary_predictions)
st.write(f"Accuracy: {accuracy:.2%}")

# Make predictions on the entire dataset
all_predictions = model(X).detach().numpy().flatten()
data['predictions'] = (all_predictions > 0.5).astype(int)

# Assign random coordinates to the markers
data['latitude'] = [random.uniform(10, 20) for _ in range(len(data))]
data['longitude'] = [random.uniform(70, 80) for _ in range(len(data))]

# Create a Streamlit app
st.title("Predicted Wetlands Map")

# Create a Folium Map centered around Karnataka
m = folium.Map(location=[15, 76], zoom_start=7)

# Create a MarkerCluster to handle multiple markers
marker_cluster = MarkerCluster().add_to(m)

# Add markers for each feature in the dataset with predicted labels
for index, row in data.iterrows():
    label = f"Predicted Label: {row['predictions']}"
    color = 'green' if row['predictions'] == 1 else 'red'
    folium.Marker(location=[row['latitude'], row['longitude']], popup=label, icon=folium.Icon(color=color)).add_to(marker_cluster)

# Display the map using folium_static
folium_static(m)
