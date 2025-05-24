import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# 物理空间参数
CELL_SIZE = 0.5  # 元胞尺寸 (米)
ROOM_WIDTH = 15.0  # X轴总长 = 30个元胞 * 0.5m
ROOM_HEIGHT = 18.0  # Y轴总长 = 36个元胞 * 0.5m

def create_heatmap_grid(df):
    """创建基于物理空间映射的热力图网格"""
    # 初始化三维网格 (y, x, [temp, co, soot])
    grid = np.full((int(ROOM_HEIGHT/CELL_SIZE), 
                   int(ROOM_WIDTH/CELL_SIZE), 3), np.nan)
    
    # 物理坐标到网格索引转换
    def coord2index(x, y):
        return (int(y//CELL_SIZE), int(x//CELL_SIZE))
    
    # 传感器数据采集 (改进型)
    sensor_points = []
    for col in df.columns:
        # 解析设备类型和编号
        if 'Device' in col:
            dev_num = int(col[6:])
            if 771 <= dev_num <= 1060:  # 修正设备号范围
                # 计算物理位置 (右上区域)
                rel_x = (dev_num - 771) % 15 * CELL_SIZE + ROOM_WIDTH/2
                rel_y = (dev_num - 771) // 15 * CELL_SIZE + ROOM_HEIGHT/2
                sensor_points.append( (rel_x, rel_y, 'temp') )
        
        elif col.startswith('CO'):
            co_num = int(col[2:]) if col[2:].isdigit() else 0
            if 1 <= co_num <= 530:  # 添加有效性检查
                x = (co_num % 18) * CELL_SIZE * 2 
                y = (co_num // 18) * CELL_SIZE
                sensor_points.append( (x, y, 'co') )
            
        elif col.startswith('soot'):
            soot_num = max(1, int(col[4:]))  # 防止负值
            if 1 <= soot_num <= 271:
                x = ROOM_WIDTH/2 + (soot_num % 15) * CELL_SIZE
                y = (soot_num // 15) * CELL_SIZE
                sensor_points.append( (x, y, 'soot') )
    
    # 空间插值处理 (改进版)
    for layer, param in enumerate(['temp', 'co', 'soot']):
        # 提取有效数据点
        param_points = [p for p in sensor_points if p[2] == param]
        if not param_points:
            print(f"警告：{param}传感器数据缺失，使用默认值填充")
            grid[:, :, layer] = 0
            continue
            
        points = np.array([[x, y] for x, y, _ in param_points])
        values = np.array([df[f"{param}{num}"].mean() for (x, y, _), num in param_points])
        
        # 生成插值网格
        xi = np.linspace(0, ROOM_WIDTH, grid.shape[1])
        yi = np.linspace(0, ROOM_HEIGHT, grid.shape[0])
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        xi_points = np.column_stack((xi_grid.ravel(), yi_grid.ravel()))

        try:
            # 执行插值
            interpolated = griddata(points, values, xi_points, 
                                   method='linear', fill_value=0)
            grid[:, :, layer] = interpolated.reshape(grid.shape[:2])
        except ValueError as e:
            print(f"{param}插值失败：{e}")
            grid[:, :, layer] = 0

    # 对称性优化 (四向镜像平均)
    def symmetry_enhance(matrix):
        return (matrix + np.flipud(matrix) + np.fliplr(matrix) + np.fliplr(np.flipud(matrix))) /4
    
    for i in range(3):
        grid[:,:,i] = symmetry_enhance(grid[:,:,i])
    
    return grid

def plot_enhanced_heatmaps(grid):
    """增强型热力图可视化"""
    plt.style.use('seaborn-dark')
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'Arial',
        'axes.labelcolor': '#404040',
        'axes.edgecolor': '#808080'
    })
    
    fig, axs = plt.subplots(1, 3, figsize=(24, 8), 
                          gridspec_kw={'wspace':0.25})
    
    # 可视化参数配置
    configs = [
        {'title':'Temperature Distribution', 
         'cmap':'YlOrRd', 'unit':'°C', 'vmax':300},
        {'title':'CO Concentration', 
         'cmap':'Blues', 'unit':'ppm', 'vmax':100},
        {'title':'Soot Concentration',
         'cmap':'Greys', 'unit':'μg/m³', 'vmax':50}
    ]
    
    for ax, cfg in zip(axs, configs):
        layer = configs.index(cfg)
        im = ax.imshow(grid[:,:,layer], 
                      cmap=cfg['cmap'],
                      extent=[0, ROOM_WIDTH, 0, ROOM_HEIGHT],
                      origin='lower', 
                      vmin=0, 
                      vmax=cfg['vmax'],
                      aspect='equal')  # 保持1:1比例
        
        # 添加物理空间标注
        ax.set_xticks(np.arange(0, ROOM_WIDTH+1, 3))
        ax.set_yticks(np.arange(0, ROOM_HEIGHT+1, 3))
        ax.set_xlabel('X (meters)', fontweight='semibold')
        ax.set_ylabel('Y (meters)', fontweight='semibold')
        ax.set_title(cfg['title'], fontsize=14, pad=15)
        
        # 添加元胞网格线
        ax.grid(which='both', color='#B0B0B0', linestyle=':', linewidth=0.4)
        
        # 添加色标
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(cfg['unit'], rotation=270, 
                      labelpad=20, fontweight='medium')
    
    # 保存高分辨率图像
    plt.savefig('enhanced_heatmaps.png', dpi=300, 
               bbox_inches='tight', facecolor='white')
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv(r'D:\github\CA\Louvre_Evacuation-master\code\robo_devc.csv')
    if df is not None:
        grid = create_heatmap_grid(df)
        plot_enhanced_heatmaps(grid)
        np.save('optimized_grid.npy', grid)