import csv
from matplotlib import pyplot as plt

# 文件路径（使用原始字符串）
file1 = r"C:\Users\Administrator\Desktop\CA&robot\code\Louvre_Evacuation-master\code\picture\Session 3.csv"  # 请替换为实际路径
file2 = r"C:\Users\Administrator\Desktop\CA&robot\code\Louvre_Evacuation-master\code\picture\Session 5.csv"

# 读取Session 3数据
velocity1, distance1, time1 = [], [], []
with open(file1, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # 跳过标题行
    for row in reader:
        try:
            # A列是时间（索引0）
            # V列是速度（索引21）
            # U列是距离（索引20）
            time1.append(float(row[0]))     # A列
            velocity1.append(float(row[21]))  # V列
            distance1.append(float(row[20]))  # U列
        except (ValueError, IndexError) as e:
            print(f"跳过无效行: {e}")

# 读取Session 5数据（列索引相同）
velocity2, distance2, time2 = [], [], []
with open(file2, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        try:
            current_time = float(row[0])  # A列时间
            if not (1860 <= current_time <= 1900):
                time2.append(current_time)
                velocity2.append(float(row[21]))  # V列速度
                distance2.append(float(row[20]))  # U列距离
        except (ValueError, IndexError) as e:
            print(f"跳过无效行: {e}")

# 创建图表
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# 绘制图表（保持原有绘图代码不变）
ax1.plot(time1, velocity1, 'b-')
ax1.set_title('Session 3: Time-Velocity')
ax1.set_xlabel('Time')
ax1.set_ylabel('Velocity (km/h)')

ax2.plot(distance1, velocity1, 'r-')
ax2.set_title('Session 3: Distance-Velocity')
ax2.set_xlabel('Distance (m)')
ax2.set_ylabel('Velocity (km/h)')

ax3.plot(time2, velocity2, 'b-')
ax3.set_title('Session 5: Time-Velocity')
ax3.set_xlabel('Time')
ax3.set_ylabel('Velocity (km/h)')

ax4.plot(distance2, velocity2, 'r-')
ax4.set_title('Session 5: Distance-Velocity')
ax4.set_xlabel('Distance (m)')
ax4.set_ylabel('Velocity (km/h)')

plt.tight_layout()
plt.show()