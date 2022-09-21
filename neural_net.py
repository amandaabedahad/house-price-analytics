import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns


class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


class Simple_nn(nn.Module):
    def __init__(self, input_features_dim, output_dim, hidden_dim):
        super(Simple_nn, self).__init__()
        self.fc1 = nn.Linear(input_features_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.output(x)
        return x


if __name__ == "__main__":
    batch_size = 16
    nr_epochs = 500
    loss_values = []
    percentage_off_values = []

    hemnet_house_data = pd.read_csv("hemnet_data/hemnet_house_data_processed.csv")
    filtered_data = hemnet_house_data[hemnet_house_data["housing_type"] == "Lägenhet"]
    data = filtered_data[["sqr_meter", "nr_rooms", "final_price", "latitude", "longitude"]].dropna()
    y = data["final_price"]
    X = data.drop(columns=["final_price"])
    std_X = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.33, random_state=42)
    sns.pairplot(X)
    X_train = std_X.fit_transform(X_train)
    X_test = std_X.transform(X_test)

    train_data = Data(X_train, y_train)
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    test_data = Data(X_test, y_test)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    net = Simple_nn(input_features_dim=4, output_dim=1, hidden_dim=32)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    for epoch in range(nr_epochs):
        for X, y in train_dataloader:
            optimizer.zero_grad()
            pred = net(X)
            loss = criterion(pred, torch.reshape(y, (-1, 1)))
            loss_values.append(loss.item())
            percentage_off = y/pred
            percentage_off_values.append(percentage_off.mean().item())
            loss.backward()
            optimizer.step()

    step = np.linspace(0, nr_epochs, 91000)

    fig, ax = plt.subplots(figsize=(8, 5))
    plt.plot(step, np.array(percentage_off_values))
    plt.title("Step-wise Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

