import numpy as np
import math
import pandas as pd

def load_field_from_csv(csv_path, width, height):
    """
    width, height: 全场宽高
    返回: shape=(height, width) 的矩阵
    """
    df = pd.read_csv(csv_path)
    # 去掉时间列
    data = df.drop(columns=['Time'], errors='ignore').values.flatten()
    # 排序
    data_sorted = np.sort(data)

    h_half = height // 2
    w_half = width // 2
    quarter_size = h_half * w_half

    quarter = data_sorted[:quarter_size].reshape((h_half, w_half))

    left_half = np.concatenate([quarter, np.fliplr(quarter)], axis=1)

    full_field = np.concatenate([left_half, np.flipud(left_half)], axis=0)

    if full_field.shape[0] < height:
        last_row = full_field[-1:,:]
        full_field = np.vstack([full_field, last_row])
    if full_field.shape[1] < width:
        last_col = full_field[:,-1:]
        full_field = np.hstack([full_field, last_col])
    # 裁剪到目标shape
    return full_field[:height, :width]

class FireSource:
    """火灾物理模型"""
    def __init__(self, center, size, temp_max=600, co_max=1500):
        """
        参数:
        - center: (x,y) 火源中心坐标
        - size: (width,height) 火源物理尺寸
        - temp_max: 最高温度(℃)
        - co_max: 最大CO浓度(ppm)
        """
        self.center = np.array(center)
        self.radius = max(size) * 1.5  # 影响半径是物理尺寸的1.5倍
        self.temp_max = temp_max
        self.co_max = co_max
        self.size = np.array(size)
        
    def get_temperature(self, pos):
        """计算位置的温度"""
        distance = np.linalg.norm(np.array(pos) - self.center)
        if self.radius <= 0:
            return 0.0
        return self.temp_max * max(0, 1 - distance/self.radius)
    
    def get_co_concentration(self, pos):
        """计算位置的CO浓度"""
        distance = np.linalg.norm(np.array(pos) - self.center)
        if self.radius <= 0:
            return 0.0
        return self.co_max * max(0, 1 - distance/self.radius)
    
    def get_visibility(self, pos):
        """计算能见度(0-1)"""
        distance = np.linalg.norm(np.array(pos) - self.center)
        if self.radius <= 0:
            return 1.0  # 无火源时能见度最好
        norm_dist = distance / self.radius
        return 1 / (1 + math.exp(-10*(norm_dist-0.5)))  # S型曲线
    
    def get_danger_level(self, pos):
        """增强版危险度计算"""
        distance = np.linalg.norm(np.array(pos) - self.center)
        if distance <= 0.5:  # 火源核心区域
            return 1.0
        
        # 防止除以零
        if self.radius <= 0:
            return 0.0
        
        # 计算各项指标
        temp_norm = self.get_temperature(pos) / self.temp_max
        co_norm = self.get_co_concentration(pos) / self.co_max
        visibility = self.get_visibility(pos)
        
        # 指数衰减危险度（更陡峭）
        distance_effect = math.exp(-2 * distance/self.radius)
        
        # 添加温度梯度影响
        temp_effect = min(1.0, 0.6 + 0.4/(1 + distance))
        
        # 综合危险度计算 - 加大CO和烟雾影响
        danger = (
            0.3 * distance_effect +      # 距离影响
            0.2 * temp_effect +          # 温度影响
            0.2 * temp_norm +            # 温度标准化
            0.3 * co_norm                # CO浓度影响加大到0.3
        )
        
        return min(danger, 1.0)  # 确保危险度不超过1.0

class FireSpreadModel:
    """火灾蔓延模型"""
    def __init__(self, sources, width=18, height=15):
        self.sources = sources
        self.width = width
        self.height = height
        # 加载CO、soot、temperature场
        self.co_field = load_field_from_csv('code/co_data.csv', width, height)
        self.soot_field = load_field_from_csv('code/soot_data.csv', width, height)
        self.temp_field = load_field_from_csv('code/temperature_data.csv', width, height)
        
    def get_max_danger(self, pos):
        """获取所有火源中的最大危险度"""
        return max(source.get_danger_level(pos) for source in self.sources)
    
    def get_temperature_field(self, grid_size):
        """生成温度场矩阵"""
        width, height = grid_size
        temp_field = np.zeros((width, height))
        
        for i in range(width):
            for j in range(height):
                pos = (i, j)
                # 计算所有火源在该点的温度总和
                temp_field[i, j] = sum(source.get_temperature(pos) for source in self.sources)
        
        return temp_field

    def get_co_concentration(self, pos):
        x, y = int(pos[0]), int(pos[1])
        if 0 <= y < self.height and 0 <= x < self.width:
            return self.co_field[y, x]
        return 0

    def get_soot_concentration(self, pos):
        x, y = int(pos[0]), int(pos[1])
        if 0 <= y < self.height and 0 <= x < self.width:
            return self.soot_field[y, x]
        return 0

    def get_temperature(self, pos):
        x, y = int(pos[0]), int(pos[1])
        if 0 <= y < self.height and 0 <= x < self.width:
            return self.temp_field[y, x]
        return 0