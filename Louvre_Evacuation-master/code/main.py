import tkinter as tk
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import time
import random

from map import myMap
from people import People

class GUI:
    Pic_Ratio = 20  # 每个格子20像素(0.5m×20=10px)
    

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("疏散模拟")
        self.root.geometry("1000x715")
        self.root.resizable(width=True, height=True)

        width = myMap.Length * GUI.Pic_Ratio
        height = myMap.Width * GUI.Pic_Ratio
        self.canvas = tk.Canvas(self.root, width=width, height=height, bg="grey")
        self.canvas.pack()

        self.label_time = tk.Label(self.root, text="Time = 0.00s", font='Arial -37 bold')
        self.label_evac = tk.Label(self.root, text="Evacution People: 0", font='Arial -37 bold')
        self.label_time.pack()
        self.label_evac.pack()

        self.setBarrier()
        self.setExit()

    # 障碍
    def setBarrier(self):
        for (A, B) in myMap.Barrier:
            x1, y1, x2, y2 = A[0], A[1], B[0], B[1]
            [x1, y1, x2, y2] = map(lambda x:x*GUI.Pic_Ratio, [x1, y1, x2, y2])
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="red", outline="red")
    
    # 出口
    def setExit(self):
        for (x, y) in myMap.Exit:
            sx, sy = x-1, y-1
            ex, ey = x+1, y+1
            [sx, sy, ex, ey] = map(lambda x:x*GUI.Pic_Ratio, [sx, sy, ex, ey])
            self.canvas.create_rectangle(sx, sy, ex, ey, fill="green", outline="green")

    def Update_People(self, People_List):
        for p in People_List:
            self.canvas.delete(p.name())
        self.Show_People(People_List)
    
    def Show_People(self, People_List):
        self.canvas.delete("all")  # 清空画布
        self.setBarrier()  # 重绘障碍物
        self.setExit()     # 重绘出口
        
        # 绘制机器人
        rx, ry = myMap.robot_position
        x1 = rx * self.Pic_Ratio
        y1 = ry * self.Pic_Ratio
        x2 = (rx + 1) * self.Pic_Ratio
        y2 = (ry + 1) * self.Pic_Ratio
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="blue", outline="black", tag="robot")
        
        # 绘制人群
        for p in People_List:
            if p.savety:
                continue
            # 绘制占据整个格子
            x, y = int(p.pos[0]), int(p.pos[1])  # 直接使用整数格子坐标
            x1 = x * self.Pic_Ratio
            y1 = y * self.Pic_Ratio
            x2 = (x + 1) * self.Pic_Ratio
            y2 = (y + 1) * self.Pic_Ratio
            self.canvas.create_rectangle(x1, y1, x2, y2, 
                                      fill="black", 
                                      outline="gray", 
                                      tag=p.name())
        
        # 确保机器人始终在最上层显示
        self.canvas.tag_raise("robot")

def Cellular_Automata(Total_People):
    UI = GUI()
    P = People(Total_People, myMap)
    UI.Show_People(P.list)

    Time_Start = time.time()
    Eva_Number = 0
    while Eva_Number < Total_People:
        # 检查机器人周围是否有人
        has_people_nearby = False
        for p in P.list:
            if not p.savety:  # 只检查未疏散的人
                dist_to_robot = np.linalg.norm(np.array(p.pos) - np.array(myMap.robot_position))
                if dist_to_robot <= 2:  # 如果人在机器人周围两格内
                    has_people_nearby = True
                    break
        
        # 只有当周围有人时才移动机器人
        if has_people_nearby:
            myMap.move_robot()
            
        # 更新人群位置，确保不会与机器人重叠
        for p in P.list:
            if not p.savety:
                # 检查新位置是否与机器人重叠
                if tuple(p.pos) == tuple(myMap.robot_position):
                    # 如果重叠，尝试移动到相邻的空位置
                    for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                        new_x = int(p.pos[0]) + dx
                        new_y = int(p.pos[1]) + dy
                        if myMap.Check_Valid(new_x, new_y) and (new_x, new_y) != tuple(myMap.robot_position):
                            p.pos = [new_x, new_y]
                            break
        
        Eva_Number = P.run()
        UI.Update_People(P.list)    
        time.sleep(0.1)
        UI.canvas.update()
        UI.root.update()

        Time_Pass = time.time()-Time_Start
        UI.label_time['text'] = "Time = "+ "%.2f" % Time_Pass + "s"
        UI.label_evac['text'] = "Evacution People: " + str(Eva_Number)

    Time_Pass = time.time()-Time_Start
    UI.label_time['text'] = "Time = "+  "%.2f" % Time_Pass + "s"
    UI.label_evac['text'] = "Evacution People: " + str(Eva_Number)

    # 热力图
    plt.figure(figsize=(10, 8))
    plt.imshow(P.thmap.T, cmap='Reds')
    plt.colorbar(label='Density')
    plt.title('Evacuation Heat Map')
    plt.axis('equal')
    plt.savefig('evacuation_heatmap.png')
    plt.show()

    UI.root.mainloop()


Cellular_Automata(Total_People=150)  # 减少人数以适应格子限制

