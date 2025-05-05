import numpy as np
from queue import Queue
import random
from fire_model import FireSource, FireSpreadModel  # 新增导入
import heapq
Direction = {
    "RIGHT": 0, "UP": 1, "LEFT": 2, "DOWN": 3, "NONE": -1
}

MoveTO = []
MoveTO.append(np.array([1, 0]))     # RIGHT
MoveTO.append(np.array([0, -1]))    # UP
MoveTO.append(np.array([-1, 0]))    # LEFT
MoveTO.append(np.array([0, 1]))     # DOWN
MoveTO.append(np.array([1, -1]))    
MoveTO.append(np.array([-1, -1]))   
MoveTO.append(np.array([-1, 1]))    
MoveTO.append(np.array([1, 1]))     


def Init_Exit(P1, P2):
    exit = list()
    
    if P1[0] == P2[0]:
        x = P1[0]
        for y in range(P1[1], P2[1]+1):
            exit.append((x, y))
    elif P1[1] == P2[1]:
        y = P1[1]
        for x in range(P1[0], P2[0]+1):
            exit.append((x, y))
    else:
        pass

    return exit

def Init_Barrier(A, B):
    if A[0] > B[0]:
        A, B = B, A

    x1, y1 = A[0], A[1]
    x2, y2 = B[0], B[1]

    if y1 < y2:
        return ((x1, y1), (x2, y2))
    else:
        return ((x1, y2), (x2, y1))

Outer_Size = 1

class Map:
    def __init__(self, L, W, E, B):
        self.Length = L
        self.Width = W
        self.Exit = E
        self.Barrier = B
        self.barrier_list = []
        self.space = np.zeros((self.Length+Outer_Size*2, self.Width+Outer_Size*2))
        self.static_field = self._calculate_static_field()

        
        for j in range(0, self.Width+Outer_Size*2):
            self.space[0][j] = self.space[L+1][j] = float("inf")
            self.barrier_list.append((0, j))
            self.barrier_list.append((L+1, j))

        for i in range(0, self.Length+Outer_Size*2):
            self.space[i][0] = self.space[i][W+1] = float("inf")
            self.barrier_list.append((i, 0))
            self.barrier_list.append((i, W+1))

        for (A, B) in self.Barrier:
            for i in range(A[0], B[0]+1):
                for j in range(A[1], B[1]+1):
                    self.space[i][j] = float("inf")
                    self.barrier_list.append((i, j))
        self.fire_model = FireSpreadModel([
            FireSource(
                center=((A[0]+B[0])/2, (A[1]+B[1])/2),
                size=(abs(B[0]-A[0]), abs(B[1]-A[1])),
                temp_max=800,
                co_max=2000
            ) for (A,B) in self.Barrier
        ])
        
        for (ex, ey) in self.Exit:
            self.space[ex][ey] = 1
            if ex == self.Length:
                self.space[ex+1][ey] = 1
            if ey == self.Width:
                self.space[ex][ey+1] = 1
            if (ex, ey) in self.barrier_list:
                self.barrier_list.remove((ex, ey))
        
        self.Init_Potential()

    def print(self, mat):
        for line in mat:
            for v in line:
                print(v, end=' ')
            print("")
    
    def Check_Valid(self, x, y):
        x, y = int(x), int(y)
        if x > self.Length+1 or x < 0 or y > self.Width+1 or y < 0:
            return False
        if self.space[x][y] == float("inf"):
            return False
        else:
            return True

    def checkSavefy(self, pos):
        x, y = int(pos[0]), int(pos[1])
        if x == self.Length+1:
            x -= 1
        elif x == -1:
            x += 1
        if y == self.Width+1:
            y -= 1
        elif y == -1:
            y -= 0

        if (x, y) in self.Exit:
            return True
        else:
            return False

    def getDeltaP(self, P1, P2):
        x1, y1 = int(P1[0]), int(P1[1])
        x2, y2 = int(P2[0]), int(P2[1])
        return self.space[x1][y1] - self.space[x2][y2]


    def _init_fire_sources(self):
        self.fire_model = FireSpreadModel([
            FireSource(
                center=((A[0]+B[0])/2, (A[1]+B[1])/2),
                size=(abs(B[0]-A[0]), abs(B[1]-A[1])),
                temp_max=800,
                co_max=2000
            ) for (A,B) in self.Barrier
        ])
        
    def get_fire_danger(self, pos):
        if hasattr(self, 'fire_model'):
            return self.fire_model.get_max_danger(pos)
        return 0.0

    def Init_Potential(self):
        minDis = np.zeros((self.Length+Outer_Size*2, self.Width+Outer_Size*2))
        for i in range(self.Length+Outer_Size*2):
            for j in range(self.Width+Outer_Size*2):
                danger = self.get_fire_danger((i,j))
                fire_penalyt = 200*(danger ** 2)
                minDis[i][j] = float("inf")

        for (sx, sy) in self.Exit:
            tmp = self.BFS(sx, sy)
            for i in range(self.Length+Outer_Size*2):
                for j in range(self.Width+Outer_Size*2):
                    minDis[i][j] = min(minDis[i][j], tmp[i][j])

        self.space = minDis

    def BFS(self, x, y):
        if not self.Check_Valid(x, y):
            return

        tmpDis = np.zeros((self.Length+Outer_Size*2, self.Width+Outer_Size*2))
        for i in range(self.Length+Outer_Size*2):
            for j in range(self.Width+Outer_Size*2):
                tmpDis[i][j] = self.space[i][j]

        queue = Queue()
        queue.put((x, y))
        tmpDis[x][y] = 1
        while not queue.empty():
            (x, y) = queue.get()
            dis = tmpDis[x][y]
            for i in range(8):
                move = MoveTO[i]
                (nx, ny) = (x, y) + move
                if self.Check_Valid(nx, ny) and tmpDis[nx][ny] == 0:
                    queue.put((nx, ny))
                    tmpDis[nx][ny] = dis + (1.0 if i < 4 else 1.4)

        return tmpDis
    def _calculate_static_field(self):
        """使用Dijkstra算法计算静态场"""
        field = np.full((self.Length+2, self.Width+2), float('inf'))
        heap = []
        
        # 初始化出口距离为0
        for (ex, ey) in self.Exit:
            field[ex][ey] = 0
            heapq.heappush(heap, (0, ex, ey))
        
        # Dijkstra算法
        moves = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1), (1,-1), (-1,-1)]
        while heap:
            dist, x, y = heapq.heappop(heap)
            
            for dx, dy in moves:
                nx, ny = x + dx, y + dy
                if not self.Check_Valid(nx, ny):
                    continue
                    
                # 对角线移动距离为√2，其他为1
                step = 1.4 if dx != 0 and dy != 0 else 1.0
                new_dist = dist + step
                
                if new_dist < field[nx][ny]:
                    field[nx][ny] = new_dist
                    heapq.heappush(heap, (new_dist, nx, ny))
        
        # 归一化处理
        max_val = np.max(field[field != float('inf')])
        field[field == float('inf')] = max_val * 2  # 障碍物区域赋予较大值
        return max_val - field  # 转换为吸引力场（值越大吸引力越强）
    def Random_Valid_Point(self):
        x = random.uniform(1, self.Length+2)
        y = random.uniform(1, self.Width+2)
        while not myMap.Check_Valid(x, y):
            x = random.uniform(1, self.Length+2)
            y = random.uniform(1, self.Width+2)
        return x, y

def Init_Map():
    Length = 36
    Width = 30
    Exit = Init_Exit(P1=(36, 14), P2=(36, 16))
    Barrier = [
        Init_Barrier(A=(18, 14), B=(20, 16))
    ]
    return Map(L=Length, W=Width, E=Exit, B=Barrier)

Length = 36
Width = 30
Exit = Init_Exit(P1=(36, 14), P2=(36, 16))
Barrier = [
    Init_Barrier(A=(18, 14), B=(20, 16))
]
myMap = Map(L=Length, W=Width, E=Exit, B=Barrier)
