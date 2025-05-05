import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def process_device_data(file_path="code/robo_devc.csv"):

    try:
        # 1. 加载CSV文件(跳过首行单位说明)
        df = pd.read_csv(file_path, skiprows=1)
        
        # 2. 列名处理
        df.columns = [col.strip() for col in df.columns]
        
        # 3. 转换科学计数法数值
        numeric_cols = df.columns[df.columns != 'Device']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        # 4. 设置时间索引
        df = df.rename(columns={'Time': 'time'})
        df['time'] = pd.to_numeric(df['time'], errors='coerce')
        df.set_index('time', inplace=True)
        
        # 5. 设备状态数据提取
        device_cols = [col for col in df.columns if col.startswith('Device')]
        device_data = df[device_cols]
        
        # 6. 保存处理结果
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        
        # 保存CSV
        device_data.to_csv(output_dir / 'processed_device_data.csv')
        
        # 绘制趋势图
        plt.figure(figsize=(12, 6))
        for col in device_cols[:5]:  # 只绘制前5个设备避免图表混乱
            plt.plot(device_data.index, device_data[col], label=col)
        plt.legend()
        plt.savefig(output_dir / 'device_trends.png')
        
        print(f"处理完成！数据已保存到 {output_dir} 目录")
        return device_data
    
    except Exception as e:
        print(f"处理出错: {str(e)}")
        return None

# 使用示例
if __name__ == "__main__":
    data = process_device_data()
    if data is not None:
        print("\n数据摘要:")
        print(data.describe())
        
        print("\n前5个时间点的数据:")
        print(data.head())