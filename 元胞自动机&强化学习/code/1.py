import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 数据读取（请替换为实际文件路径）
file_path = r"C:\Users\Administrator\Desktop\CA&robot\code\Louvre_Evacuation-master\code\robo_devc.csv"  # 支持csv/excel等格式
try:
    df = pd.read_csv(file_path, encoding='utf-8')  # 或 pd.read_excel()
except:
    df = pd.read_csv(file_path, encoding='gbk')  # 尝试中文编码

# 2. 数据预览
print("数据前5行：")
print(df.head())
print("\n数据统计描述：")
print(df.describe())
print("\n缺失值检查：")
print(df.isnull().sum())

# 3. 数据清洗
# 处理缺失值（根据实际情况选择）
df.fillna(method='ffill', inplace=True)  # 前向填充
# 或 df.dropna(inplace=True)  # 删除缺失值

# 4. 单变量分析
plt.figure(figsize=(15, 4))

# CO浓度分布
plt.subplot(131)
sns.histplot(df['CO浓度'], kde=True, bins=30)
plt.title('CO浓度分布')

# 温度分布
plt.subplot(132)
sns.histplot(df['温度'], kde=True, bins=30)
plt.title('温度分布')

# 烟气浓度分布
plt.subplot(133)
sns.histplot(df['烟气浓度'], kde=True, bins=30)
plt.title('烟气浓度分布')
plt.tight_layout()
plt.show()

# 5. 多变量关系分析
sns.pairplot(df[['CO浓度', '温度', '烟气浓度']])
plt.suptitle('变量间关系矩阵', y=1.02)
plt.show()

# 6. 时间序列分析（如果有时间列）
if '时间' in df.columns:
    plt.figure(figsize=(12, 6))
    df.set_index('时间')[['CO浓度', '温度', '烟气浓度']].plot(subplots=True)
    plt.title('时间序列变化')
    plt.tight_layout()
    plt.show()

# 7. 相关性分析
corr_matrix = df[['CO浓度', '温度', '烟气浓度']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('变量相关性热力图')
plt.show()