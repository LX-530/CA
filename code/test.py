import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import re

# 物理空间参数
CELL_SIZE = 0.5  # 元胞尺寸 (米)
ROOM_WIDTH = 15.0  # X轴总长 = 30个元胞 * 0.5m
ROOM_HEIGHT = 18.0  # Y轴总长 = 36个元胞 * 0.5m

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def visualize_quarter_csv(csv_path, width, height, scale=1.0, cmap='hot', title='', use_last_row=False):
    """
    读取左上四分之一数据，镜像补全全场并可视化
    :param csv_path: 数据文件路径
    :param width: 全场宽
    :param height: 全场高
    :param scale: 缩放系数
    :param cmap: 热力图色带
    :param title: 图标题
    :param use_last_row: 若为True，则只用最后一行数据
    """
    df = pd.read_csv(csv_path)
    if use_last_row:
        # 只用最后一行
        row = df.iloc[-1]
        if 'Time' in row.index:
            data = row.drop(labels=['Time']).values.astype(float)
        else:
            data = row.values.astype(float)
    else:
        # 默认全量
        if 'Time' in df.columns:
            data = df.drop(columns=['Time'], errors='ignore').values.flatten()
        else:
            data = df.values.flatten()
    # 按数值从小到大排序
    data_sorted = np.sort(data)
    h_half = height // 2
    w_half = width // 2
    quarter_size = h_half * w_half
    if data_sorted.shape[0] < quarter_size:
        raise ValueError("数据量不足以填满左上四分之一区域")
    quarter = data_sorted[:quarter_size].reshape((h_half, w_half))
    left_half = np.concatenate([quarter, np.fliplr(quarter)], axis=1)
    full_field = np.concatenate([left_half, np.flipud(left_half)], axis=0)
    if full_field.shape[0] < height:
        last_row = full_field[-1:,:]
        full_field = np.vstack([full_field, last_row])
    if full_field.shape[1] < width:
        last_col = full_field[:,-1:]
        full_field = np.hstack([full_field, last_col])
    full_field = full_field[:height, :width]
    full_field = full_field * scale

    print("min:", full_field.min(), "max:", full_field.max())
    plt.figure(figsize=(8, 6))
    plt.imshow(full_field, cmap=cmap, origin='lower', aspect='auto')
    plt.colorbar(label='Value (after scaling)')
    plt.title(title or f'Heatmap of {csv_path} (last row)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    plt.show()
    return full_field

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
    
    # 调整数量级
    grid[:,:,0] = grid[:,:,0] / 500000    # 温度缩小5倍
    grid[:,:,1] = grid[:,:,1] * 1000000   # CO放大10倍
    grid[:,:,2] = grid[:,:,2] * 1000000  # 烟雾放大100倍
    
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
    # 可视化CO浓度最后一行
    visualize_quarter_csv(
        'D:/github/CA/Louvre_Evacuation-master/co_data.csv', width=18, height=15, scale=1e5,
        cmap='Blues', title='CO浓度分布（最后一帧）', use_last_row=True
    )
    # 可视化温度最后一行
    visualize_quarter_csv(
        'D:/github/CA/Louvre_Evacuation-master/temperature_data.csv', width=18, height=15, scale=1,
        cmap='hot', title='温度分布（最后一帧）', use_last_row=True
    )
    # 可视化烟雾浓度最后一行
    visualize_quarter_csv(
        'D:/github/CA/Louvre_Evacuation-master/soot_data.csv', width=18, height=15, scale=1e6,
        cmap='Greys', title='烟雾分布（最后一帧）', use_last_row=True
    )