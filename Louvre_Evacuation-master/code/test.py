import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import re

# 物理空间参数
CELL_SIZE = 0.5  # 元胞尺寸 (米)
ROOM_WIDTH = 15.0  # X轴总长 = 30个元胞 * 0.5m
ROOM_HEIGHT = 18.0  # Y轴总长 = 36个元胞 * 0.5m

def read_and_categorize_data(file_path):
    """读取并分类CSV数据"""
    try:
        df = pd.read_csv(file_path)
        print("数据读取成功，前5行示例：")
        print(df.head())
        return df
    except Exception as e:
        print(f"数据读取错误: {e}")
        return None

def create_heatmap_grid(df):
    """创建基于物理空间映射的热力图网格"""
    # 初始化三维网格 (y, x, [temp, co, soot])
    grid = np.zeros((int(ROOM_HEIGHT/CELL_SIZE), 
                   int(ROOM_WIDTH/CELL_SIZE), 3))
    
    # 物理坐标到网格索引转换
    def coord2index(x, y):
        return (int(y//CELL_SIZE), int(x//CELL_SIZE))
    
    # 填充网格数据
    for i in range(grid.shape[0]):  # y轴
        for j in range(grid.shape[1]):  # x轴
            # 计算物理坐标
            x = j * CELL_SIZE
            y = i * CELL_SIZE
            
            # 温度传感器(右上区域)
            if x >= ROOM_WIDTH/2 and y >= ROOM_HEIGHT/2:
                dev_idx = 771 + coord2index(x-ROOM_WIDTH/2, y-ROOM_HEIGHT/2)[0]*15 + coord2index(x-ROOM_WIDTH/2, y-ROOM_HEIGHT/2)[1]
                col_name = f"Device{dev_idx}"
                grid[i,j,0] = df[col_name].mean() if col_name in df.columns else 20.0
            
            # CO传感器(左半区域)
            if x < ROOM_WIDTH/2:
                co_idx = 1 + i*9 + (j % 18)
                co_idx = min(co_idx, 530)
                col_name = f"CO{co_idx:02d}" if co_idx < 100 else f"CO{co_idx}"
                grid[i,j,1] = df[col_name].mean() if col_name in df.columns else 0.0
            
            # 烟尘传感器(右下区域)
            if y < ROOM_HEIGHT/2 and x >= ROOM_WIDTH/2:
                soot_idx = 1 + i*8 + (j - int(ROOM_WIDTH/2/CELL_SIZE))
                soot_idx = min(soot_idx, 271)
                col_name = f"soot{soot_idx:02d}" if soot_idx < 100 else f"soot{soot_idx}"
                grid[i,j,2] = df[col_name].mean() if col_name in df.columns else 0.0
    
    # 对称性补全
    temp_grid = grid[:, :, 0]
    temp_grid = np.maximum(temp_grid, temp_grid[::-1, ::-1])
    grid[:, :, 0] = temp_grid
    
    co_grid = grid[:, :, 1]
    co_grid[:, int(ROOM_WIDTH/2/CELL_SIZE):] = co_grid[:, :int(ROOM_WIDTH/2/CELL_SIZE)][:, ::-1]
    grid[:, :, 1] = (co_grid + np.fliplr(co_grid)) / 2
    
    soot_grid = grid[:, :, 2]
    soot_grid[int(ROOM_HEIGHT/2/CELL_SIZE):, :] = soot_grid[:int(ROOM_HEIGHT/2/CELL_SIZE), :][::-1, :]
    grid[:, :, 2] = (soot_grid + np.flipud(soot_grid)) / 2
    
    return grid

def plot_heatmaps(grid):
    """绘制三组热力图"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    titles = ['Temperature Distribution', 'CO Concentration', 'Soot Concentration']
    cmaps = ['hot', 'Blues', 'Greys']
    units = ['°C', 'ppm', 'mg/m³']
    
    for ax, title, cmap, unit, i in zip(axes, titles, cmaps, units, range(3)):
        im = ax.imshow(grid[:,:,i], cmap=cmap, origin='lower',
                      extent=[0, ROOM_WIDTH, 0, ROOM_HEIGHT], aspect='auto')
        fig.colorbar(im, ax=ax, label=unit)
        ax.set_title(title)
        ax.set_xticks(np.arange(0, ROOM_WIDTH+1, 3))
        ax.set_yticks(np.arange(0, ROOM_HEIGHT+1, 3))
        ax.grid(color='w', linestyle='--', linewidth=0.5, alpha=0.7)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = r'D:\github\CA\Louvre_Evacuation-master\code\robo_devc.csv'
    df = read_and_categorize_data(file_path)
    
    if df is not None:
        grid = create_heatmap_grid(df)
        plot_heatmaps(grid)
        np.save('heatmap_grid.npy', grid)
        print("热力图网格已保存为 heatmap_grid.npy")