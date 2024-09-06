import matplotlib.pyplot as plt
import numpy as np

# get current file directory
import os
dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"

print(dir_path)

with open(dir_path + "error_data.txt", "r") as f:
    error_data = f.read().splitlines()

print(error_data)

with open(dir_path + "error_du_data.txt", "r") as f:
    error_du_data = f.read().splitlines()

print(error_du_data)


plt.figure(figsize=(12, 6))
plt.title(r"Error between $\vec{u}_{t}$ and $\vec{u}_{ana,t}$ over time")
plt.xlabel("Timesteps")
plt.ylabel("Error")
x_range = np.arange(0, len(error_data))
plt.plot(x_range, error_data)
plt.show()

plt.figure(figsize=(12, 6))
plt.title(r"Error between $\frac{\partial \vec{u}_{t}}{\partial t}$ and $\frac{\partial \vec{u}_{ana,t}}{\partial t}$ over time")
plt.xlabel("Timesteps")
plt.ylabel("Error")
x_range = np.arange(0, len(error_du_data))
plt.plot(x_range, error_du_data)
plt.show()