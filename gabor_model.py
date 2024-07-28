#%% imports
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
# %% function definition
N = 28
bounds = np.array([-15, 15])
phi_real = np.array([0.0, 16.6])
x = np.linspace(bounds[0], bounds[1], N)
# %%
x.shape
# %%
x
# %%
y = np.sin(phi_real[0] + 0.06 * phi_real[1] * x) * np.exp(-(phi_real[0] + 0.06 * phi_real[1] * x)**2/(32.0))
y = y + np.random.randn(28)
# %%
y
# %%
plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("gabor model")
# %%
