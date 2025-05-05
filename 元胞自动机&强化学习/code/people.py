from map import MoveTO
import numpy as np
import time
import math
import random
import matplotlib.pyplot as plt

class Person:
    Normal_Speed = 1.25
    
    def __init__(self, id, pos_x, pos_y):
        self.id = id
        self.pos = (pos_x, pos_y)
        self.speed = Person.Normal_Speed
        self.savety = False
        self.path = [(pos_x, pos_y)]
    
    def name(self):
        return "ID_" + str(self.id)
    
    def __str__(self):
        return self.name() + " (%d, %d)" % (self.pos[0], self.pos[1])
    
        
        # 更新热力图
        self.thmap[new_x][new_y] += 1
        
        # 检查是否到达出口
        if self.map.checkSavefy(p.pos):
            p.savety = True
            self.rmap[new_x][new_y] = 0



class People:
    def __init__(self, cnt, myMap):
        self.list = []
        self.tot = cnt
        self.map = myMap
        self.rmap = np.zeros((myMap.Length+2, myMap.Width+2))  # 密度图
        self.thmap = np.zeros((myMap.Length+2, myMap.Width+2)) # 热力图
        self.dfield = np.zeros((myMap.Length+2, myMap.Width+2)) # 新增动态场
        self.last_rmap = np.zeros_like(self.rmap)  # 记录上一时间步的状态

        # 初始化人员(确保每人占据一个独立格子)
        occupied = set()
        for i in range(cnt):
            while True:
                # 分布在最左侧7.5m(15格)区域
                x = random.randint(0, 14)  # 0到14共15格
                y = random.randint(1, myMap.Width-2)
                if (x,y) not in occupied and myMap.Check_Valid(x, y):
                    occupied.add((x,y))
                    self.list.append(Person(i+1, x+0.5, y+0.5))  # 格子中心坐标
                    self.rmap[x][y] = 1
                    self.thmap[x][y] = 1
                    break

    def run(self):
        self.last_rmap = self.rmap.copy()
        cnt = 0
        move_plan = {}
        
        # 第一阶段：计划移动
        for p in self.list:
            if p.savety:
                cnt += 1
                continue
                
            x, y = int(p.pos[0]-0.5), int(p.pos[1]-0.5)  # 转换为格子坐标
            best_dir = self.find_best_direction(x, y)
            
            if best_dir is not None:
                dx, dy = MoveTO[best_dir]
                new_x, new_y = x + dx, y + dy
                if (new_x, new_y) not in move_plan:
                    move_plan[(new_x, new_y)] = []
                move_plan[(new_x, new_y)].append((p, x, y))
        
        # 第二阶段：执行移动(处理冲突)
        for target, movers in move_plan.items():
            random.shuffle(movers)  # 随机排序
            if len(movers) == 1:  # 无冲突
                p, old_x, old_y = movers[0]
                self.execute_move(p, old_x, old_y, *target)
            else:  # 有冲突，只移动第一个
                p, old_x, old_y = movers[0]
                self.execute_move(p, old_x, old_y, *target)
                # 其他人留在原地但记录热力图
                for p, _, _ in movers[1:]:
                    x, y = int(p.pos[0]-0.5), int(p.pos[1]-0.5)
                    self.thmap[x][y] += 1
        
        return cnt

    def find_best_direction(self, x, y):
        """增强版方向决策（火灾回避核心逻辑）"""
        best_dir = None
        max_score = -float('inf')
        
        current_danger = self.map.get_fire_danger((x, y))  # 当前位置危险度
        
        for dire in range(8):  # 检查8个方向
            dx, dy = MoveTO[dire]
            nx, ny = x + dx, y + dy
            
            if self.map.Check_Valid(nx, ny) and self.rmap[nx][ny] == 0:
                # 静态场吸引力（使用Dijkstra计算的值）
                static_attraction = self.map.static_field[nx][ny] 
                
                # 危险度计算（三个关键因素）
                next_danger = self.map.get_fire_danger((nx, ny))
                danger_diff = next_danger - current_danger  # 危险变化量
                absolute_danger = next_danger  # 绝对危险值
                # 新增加动态场影响因子
                dynamic_influence = 0.2 * self.dfield[nx][ny]  # 可调整系数
                # 综合评分公式（关键调整点）
                score = (
                    20 * static_attraction +                  # 基础势能
                     -30 * absolute_danger+#**2 +       # 危险区惩罚 
                    dynamic_influence +      # 动态场影响
                    random.uniform(-0.1, 0.1)  # 随机扰动
                    -10 * self.rmap[nx][ny]       
                )
                
                if score > max_score:
                    max_score = score
                    best_dir = dire
        
        return best_dir
    
    def execute_move(self, p, old_x, old_y, new_x, new_y):
        # 计算修正系数d1和d2
        d1 = 1 if self.last_rmap[new_x][new_y] == 0 and self.rmap[new_x][new_y] == 0 else 0
        d2 = 1 if self.last_rmap[new_x][new_y] == 1 and self.rmap[new_x][new_y] == 1 else 0
        exit_bonus = 2.0 if self.map.checkSavefy((new_x+0.5, new_y+0.5)) else 1.0
    
    # 更新动态场（出口区域获得更大增量）
        self.dfield[new_x][new_y] += exit_bonus * (1 + d1 + d2)

        
        # 保存当前状态到last_rmap
        self.last_rmap = self.rmap.copy()
        
        # 原有移动逻辑保持不变
        self.rmap[old_x][old_y] = 0
        self.rmap[new_x][new_y] = 1
        p.pos = (new_x + 0.5, new_y + 0.5)
        p.path.append(p.pos)
        
        # 更新热力图
        self.thmap[new_x][new_y] += 1
        
        # 检查是否到达出口
        if self.map.checkSavefy(p.pos):
            p.savety = True
            self.rmap[new_x][new_y] = 0


# Total_People = 2
# P = People(Total_People, myMap)


# Eva_Number = 0
# while Eva_Number<Total_People:
# 	Eva_Number = P.run()

	# time.sleep(0.5)