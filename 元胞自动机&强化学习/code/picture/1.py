import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
data = {
    '序号': [1, 2, 3, 4, 5, 6, 7, 8],
    '燃油消耗量△Q(mL)': [127.20, 90.69, 73.86, 67.27, 56.36, 49.17, 51.32, 47.14],
    '时间△t(s)': [183.70, 124.56, 94.65, 76.49, 62.48, 53.32, 47.16, 38.58],
    '平均车速Va(km/h)': [20, 30, 40, 50, 60, 70, 80, 90],
    '等速百公里燃油消耗量Q(L/100km)': [23.26, 18.07, 14.66, 13.55, 11.27, 9.83, 10.26, 10.31]
}
df = pd.DataFrame(data)
df.to_excel("answer.xlsx", index=False)
Va = df['平均车速Va(km/h)'].values
Q = df['等速百公里燃油消耗量Q(L/100km)'].values
coefficients = np.polyfit(Va, Q, 2)
polynomial = np.poly1d(coefficients)
Va_fit = np.linspace(min(Va), max(Va), 100)
Q_fit = polynomial(Va_fit)
plt.figure(dpi=128, figsize=(10, 6))
plt.scatter(Va, Q)
plt.plot(Va_fit, Q_fit, c='black')
plt.xlim(0, 100)
plt.ylim(0, 32)
plt.title("平均车速百公里耗油量", fontsize=10)
plt.xlabel('平均车速 - Va (km/h)', fontsize=8)
plt.ylabel('平均油耗 - Q (L/100km)', fontsize=8)
plt.tick_params(axis='both', which='major', labelsize=8)
plt.legend(fontsize=8)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()