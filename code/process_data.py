import pandas as pd
import numpy as np
import re

def process_data(input_file):
    # 读取数据，跳过单位行
    df = pd.read_csv(input_file, skiprows=1)
    
    # 提取时间列
    time_column = df.iloc[:, 0]
    
    # 获取所有列名
    all_columns = df.columns.tolist()
    
    # 创建三个数据框
    temp_df = pd.DataFrame()
    co_df = pd.DataFrame()
    soot_df = pd.DataFrame()
    
    # 添加时间列
    temp_df['Time'] = time_column
    co_df['Time'] = time_column
    soot_df['Time'] = time_column
    
    # 遍历每一列数据
    for col in all_columns:
        if col == 'Time':
            continue
            
        # 根据列名前缀分类
        if col.startswith('Device'):
            temp_df[col] = df[col]
        elif col.startswith('CO'):
            co_df[col] = df[col]
        elif col.startswith('soot'):
            soot_df[col] = df[col]
    
    # 对每个数据框的列进行排序（按列名中的数字）
    def sort_columns(df):
        cols = df.columns.tolist()
        time_col = cols.pop(0)  # 移除Time列
        sorted_cols = sorted(cols, key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)
        return [time_col] + sorted_cols
    
    # 重新排序列
    temp_df = temp_df[sort_columns(temp_df)]
    co_df = co_df[sort_columns(co_df)]
    soot_df = soot_df[sort_columns(soot_df)]
    
    # 对每个传感器的数值进行排序
    def sort_sensor_values(df):
        # 保存时间列
        time_col = df['Time']
        # 对每个传感器列进行排序
        for col in df.columns:
            if col != 'Time':
                df[col] = sorted(df[col].values)
        # 恢复时间列
        df['Time'] = time_col
        return df
    
    # 对三个数据框分别进行排序
    temp_df = sort_sensor_values(temp_df)
    co_df = sort_sensor_values(co_df)
    soot_df = sort_sensor_values(soot_df)
    
    # 保存到文件
    temp_df.to_csv('temperature_data.csv', index=False)
    co_df.to_csv('co_data.csv', index=False)
    soot_df.to_csv('soot_data.csv', index=False)
    
    # 打印数据统计信息
    print("温度数据统计：")
    print(f"数据点数量：{len(temp_df)}")
    print(f"温度传感器数量：{len(temp_df.columns)-1}")  # 减去Time列
    print("温度传感器列表：", [col for col in temp_df.columns if col != 'Time'])
    print("\n一氧化碳数据统计：")
    print(f"数据点数量：{len(co_df)}")
    print(f"CO传感器数量：{len(co_df.columns)-1}")
    print("CO传感器列表：", [col for col in co_df.columns if col != 'Time'])
    print("\n烟雾数据统计：")
    print(f"数据点数量：{len(soot_df)}")
    print(f"烟雾传感器数量：{len(soot_df.columns)-1}")
    print("烟雾传感器列表：", [col for col in soot_df.columns if col != 'Time'])

if __name__ == "__main__":
    input_file = "D:/github/CA/Louvre_Evacuation-master/code/robo_devc.csv"
    process_data(input_file) 