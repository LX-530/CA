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
        

def Cellular_Automata(Total_People):
    UI = GUI()
    P = People(Total_People, myMap)
    UI.Show_People(P.list)

    Time_Start = time.time()
    Eva_Number = 0
    while Eva_Number < Total_People:
        Eva_Number = P.run()
        UI.Update_People(P.list)    
        time.sleep(0.001)
        UI.canvas.update()
        UI.root.update()

        Time_Pass = time.time()-Time_Start
        UI.label_time['text'] = "Time = "+ "%.2f" % Time_Pass + "s"
        UI.label_evac['text'] = "Evacution People: " + str(Eva_Number)

    Time_Pass = time.time()-Time_Start
    UI.label_time['text'] = "Time = "+  "%.2f" % Time_Pass + "s"
    UI.label_evac['text'] = "Evacution People: " + str(Eva_Number)

    # 热力图
    sns.heatmap(P.thmap.T, cmap='Reds')
    plt.axis('equal')
    plt.show()

    UI.root.mainloop()


Cellular_Automata(Total_People=200)  # 减少人数以适应格子限制

