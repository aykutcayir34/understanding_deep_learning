#%% imports
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# %% function definition
N = 28
bounds = np.array([-15, 15])
phi_real = np.array([0.0, 16.6])
x = np.random.uniform(low=bounds[0], high=bounds[1], size=N)
# %%
x.shape
# %%
x
# %%
y = np.sin(phi_real[0] + 0.06 * phi_real[1] * x) * np.exp(-(phi_real[0] + 0.06 * phi_real[1] * x)**2/(32.0))
y = y + np.random.randn(28)
scaler = MinMaxScaler(feature_range=(-1, 1))
# %%
y = scaler.fit_transform(y.reshape(-1, 1))
# %%
plt.scatter(x, y)
plt.ylim(-1, 1)
plt.xlim(-15, 15)
plt.xlabel("x")
plt.ylabel("gabor model")
# %%
class GaborDataset(Dataset):
    def __init__(self, x, y) -> None:
        super().__init__()
        x = np.column_stack([np.ones_like(x), x])
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return len(self.x)
# %%
dataset = GaborDataset(x, y)
# %%
data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
# %%
class GaborModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(2, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
    def forward(self, x):
        return self.phi(x)
# %%
model = GaborModel()
# %%
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
epoch = 10000
loss_func = nn.MSELoss()
# %%
for i in range(epoch):
    for (x_i, y_i) in data_loader:
        y_pred = model(x_i.float())
        loss = loss_func(y_i.float(), y_pred)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    if i % 1000 == 0:
        print(f"Epoch: {i}, Loss: {loss.item()}")
# %%
x_test_ = np.linspace(start=-15, stop=15, num=N)
y_test = np.sin(phi_real[0] + 0.06 * phi_real[1] * x_test_) * np.exp(-(phi_real[0] + 0.06 * phi_real[1] * x_test_)**2/(32.0))
y_test = y_test + np.random.randn(28)
scaler = MinMaxScaler(feature_range=(-1, 1))
y_test = scaler.fit_transform(y_test.reshape(-1, 1))
x_test = np.column_stack([np.ones_like(x_test_), x_test_])
x_test = torch.from_numpy(x_test)
y_preds = []
for i in range(N):
    x_i = x_test[i, :]
    y_pred = model(x_i.float())
    y_preds.append(y_pred.item())
# %%
plt.plot(x_test_, y_preds)
plt.scatter(x_test_, y_test)
plt.xlim(-15, 15)
plt.ylim(-1.5, 1.5)
plt.xlabel("x")
plt.ylabel("gabor model")
# %%
