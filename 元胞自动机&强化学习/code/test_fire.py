import numpy as np
import matplotlib.pyplot as plt
from fire_model import FireSource

# 测试单个火源
fire = FireSource(center=(10,10), size=(4,4), temp_max=800, co_max=2000)

# 生成危险度热力图
grid_size = 20
x = np.arange(0, grid_size, 0.5)
y = np.arange(0, grid_size, 0.5)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

for i in range(len(x)):
    for j in range(len(y)):
        Z[j,i] = fire.get_danger_level((x[i],y[j]))

# 可视化
plt.figure(figsize=(10,8))
plt.contourf(X, Y, Z, levels=20, cmap='hot')
plt.colorbar(label='Danger Level')
plt.scatter(fire.center[0], fire.center[1], c='blue', marker='x', label='Fire Center')
plt.title('Fire Danger Distribution')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.show()