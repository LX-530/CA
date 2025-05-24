import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import time
from map import Map, Init_Exit, Init_Barrier, myMap
from fire_model import FireSource, FireSpreadModel
from people import People

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Environment definition
class EvacuationEnv:
    def __init__(self, width, height, fire_zones, exit_location, num_people):
        self.width = width
        self.height = height
        self.fire_zones = fire_zones
        self.exit_location = exit_location
        self.num_people = num_people
        self.robot_range = (8, 15)  # 机器人活动范围x=8~15
        self.time_step = 0.001  # 时间步长调整为0.001秒
        
        # 初始化地图和火灾模型
        self.map = Map(width, height, [exit_location], 
                      [Init_Barrier(A=fire_pos, B=fire_pos) for fire_pos in fire_zones])
        
        # 初始化火灾模型
        self.fire_model = FireSpreadModel([
            FireSource(
                center=fire_pos,
                size=(1, 1),
                temp_max=800,
                co_max=2000
            ) for fire_pos in fire_zones
        ])
        
        # 初始化可视化
        self.fig = plt.figure(figsize=(15, 5))
        self.ax1 = self.fig.add_subplot(131)  # 疏散过程
        self.ax2 = self.fig.add_subplot(132)  # 物理场分布
        self.ax3 = self.fig.add_subplot(133)  # 奖励曲线
        plt.ion()  # 打开交互模式
        
        self.reset()

    def reset(self):
        self.robot_position = [8, self.height // 2]  # 机器人初始在x=8
        self.robot_direction = 1  # 初始向右
        # 使用People类初始化人群
        self.people = People(self.num_people, self.map)
        self.time = 0
        return self._get_state()

    def move_robot(self):
        # 机器人在x=8~15之间来回巡逻
        if self.robot_position[0] >= self.robot_range[1]:
            self.robot_direction = -1
        elif self.robot_position[0] <= self.robot_range[0]:
            self.robot_direction = 1
        self.robot_position[0] += self.robot_direction

    def is_occupied(self, x, y):
        # 判断(x, y)是否被机器人占据
        return [x, y] == self.robot_position

    def _get_state(self):
        # 优化状态表示：包含更多物理信息
        # 1. 机器人位置 (2维)
        # 2. 最近的人员位置 (2维)
        # 3. 最近的火源位置 (2维)
        # 4. 出口位置 (2维)
        # 5. 当前疏散人数比例 (1维)
        # 6. 最近人员的危险度 (1维)
        # 7. 最近位置的CO浓度 (1维)
        # 8. 最近位置的烟雾浓度 (1维)
        # 9. 最近位置的温度 (1维)
        
        robot_pos = np.array(self.robot_position)
        
        # 找到最近的人员
        min_dist = float('inf')
        nearest_person_pos = np.array([0, 0])
        nearest_person_danger = 0.0
        for p in self.people.list:
            if not p.savety:
                dist = np.linalg.norm(np.array(p.pos) - robot_pos)
                if dist < min_dist:
                    min_dist = dist
                    nearest_person_pos = np.array(p.pos)
                    # 计算该人员的危险度
                    nearest_person_danger = self.fire_model.get_max_danger(p.pos)
        
        # 找到最近的火源
        min_dist = float('inf')
        nearest_fire_pos = np.array([0, 0])
        for fire_pos in self.fire_zones:
            dist = np.linalg.norm(np.array(fire_pos) - robot_pos)
            if dist < min_dist:
                min_dist = dist
                nearest_fire_pos = np.array(fire_pos)
        
        # 计算已疏散人数比例
        evacuated = sum(1 for p in self.people.list if p.savety)
        evacuation_ratio = evacuated / self.num_people
        
        # 获取最近位置的物理量
        nearest_co = self.fire_model.get_co_concentration(nearest_person_pos)
        nearest_soot = self.fire_model.get_soot_concentration(nearest_person_pos)
        nearest_temp = self.fire_model.get_temperature(nearest_person_pos)
        
        # 组合状态
        state = np.concatenate([
            robot_pos / np.array([self.width, self.height]),  # 归一化机器人位置
            nearest_person_pos / np.array([self.width, self.height]),  # 归一化最近人员位置
            nearest_fire_pos / np.array([self.width, self.height]),  # 归一化最近火源位置
            np.array(self.exit_location) / np.array([self.width, self.height]),  # 归一化出口位置
            [evacuation_ratio],  # 疏散比例
            [nearest_person_danger],  # 最近人员危险度
            [nearest_co / 2000],  # 归一化CO浓度
            [nearest_soot / 1000],  # 归一化烟雾浓度
            [nearest_temp / 800]  # 归一化温度
        ])
        
        return state

    def visualize(self, reward_history):
        """可视化当前状态"""
        # 清除旧图
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        # 1. 疏散过程可视化
        self.ax1.set_title('疏散过程')
        # 绘制出口
        self.ax1.plot(self.exit_location[0], self.exit_location[1], 'g^', markersize=10, label='出口')
        # 绘制火源
        for fire_pos in self.fire_zones:
            self.ax1.plot(fire_pos[0], fire_pos[1], 'r*', markersize=10)
        # 绘制机器人
        self.ax1.plot(self.robot_position[0], self.robot_position[1], 'bs', markersize=10, label='机器人')
        # 绘制人群
        for p in self.people.list:
            if not p.savety:
                self.ax1.plot(p.pos[0], p.pos[1], 'ko', markersize=5)
        self.ax1.set_xlim(0, self.width)
        self.ax1.set_ylim(0, self.height)
        self.ax1.grid(True)
        self.ax1.legend()
        
        # 2. 物理场分布可视化
        self.ax2.set_title('物理场分布')
        # 创建网格
        x = np.arange(self.width)
        y = np.arange(self.height)
        X, Y = np.meshgrid(x, y)
        
        # 计算每个点的危险度
        Z = np.zeros((self.height, self.width))
        for i in range(self.height):
            for j in range(self.width):
                Z[i, j] = self.fire_model.get_max_danger((j, i))
        
        # 绘制热力图
        im = self.ax2.imshow(Z, cmap='hot', origin='lower', extent=[0, self.width, 0, self.height])
        plt.colorbar(im, ax=self.ax2, label='危险度')
        self.ax2.grid(True)
        
        # 3. 奖励曲线
        self.ax3.set_title('训练奖励曲线')
        self.ax3.plot(reward_history, 'b-', label='总奖励')
        self.ax3.set_xlabel('Episode')
        self.ax3.set_ylabel('Total Reward')
        self.ax3.grid(True)
        self.ax3.legend()
        
        plt.tight_layout()
        plt.pause(0.01)  # 暂停一小段时间以更新图像

    def step(self, action):
        self.move_robot()  # 机器人先移动
        # Action: 0=up, 1=down, 2=left, 3=right, 4=stay
        next_pos = self.robot_position.copy()
        if action == 0 and self.robot_position[1] > 0:
            next_pos[1] -= 1
        elif action == 1 and self.robot_position[1] < self.height - 1:
            next_pos[1] += 1
        elif action == 2 and self.robot_position[0] > 0:
            next_pos[0] -= 1
        elif action == 3 and self.robot_position[0] < self.width - 1:
            next_pos[0] += 1
        # 机器人不能离开活动区间
        if self.robot_range[0] <= next_pos[0] <= self.robot_range[1]:
            self.robot_position = next_pos

        # 更新人群位置和健康值
        evac_count = self.people.run()
        
        # 计算奖励
        rewards = 0
        for p in self.people.list:
            if p.savety:  # 已疏散
                rewards += 200  # 疏散奖励
            else:
                # 计算到出口的距离
                dist_to_exit = np.linalg.norm(np.array(p.pos) - np.array(self.exit_location))
                # 计算到机器人的距离
                dist_to_robot = np.linalg.norm(np.array(p.pos) - np.array(self.robot_position))
                
                # 获取当前位置的物理量
                co_level = self.fire_model.get_co_concentration(p.pos)
                soot_level = self.fire_model.get_soot_concentration(p.pos)
                temp_level = self.fire_model.get_temperature(p.pos)
                danger_level = self.fire_model.get_max_danger(p.pos)
                
                # 如果人在火源附近
                if tuple(p.pos) in self.fire_zones:
                    rewards -= 100  # 在火源位置惩罚加大
                # 如果人离机器人太近
                elif dist_to_robot < 2:
                    rewards -= 20  # 避免人群聚集
                # 如果人离出口更近了
                elif dist_to_exit < self.width/2:
                    rewards += 10  # 鼓励向出口移动
                
                # 根据物理量计算惩罚
                rewards -= co_level * 0.1  # CO浓度惩罚
                rewards -= soot_level * 0.1  # 烟雾浓度惩罚
                rewards -= temp_level * 0.05  # 温度惩罚
                rewards -= danger_level * 50  # 危险度惩罚
        
        # 时间惩罚
        rewards -= 3  # 每步小惩罚，鼓励快速疏散
        
        self.time += self.time_step
        evacuated = sum(1 for p in self.people.list if p.savety)
        done = evacuated >= int(0.8 * self.num_people)
        return self._get_state(), rewards, done

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)  # 减小记忆容量
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995  # 加快探索率衰减
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.batch_size = 64  # 增加批量大小

    def _build_model(self):
        model = Sequential()
        model.add(Dense(16, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Random action
        act_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(act_values[0])  # Best action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        
        # 批量预测
        targets = self.model.predict(states, verbose=0)
        next_targets = self.model.predict(next_states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.amax(next_targets[i])
        
        # 批量训练
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Main Training Loop
if __name__ == "__main__":
    print("开始初始化环境...")
    # 使用main.py中定义的人数
    Total_People = 150  # 与main.py保持一致
    env = EvacuationEnv(18, 15, fire_zones={(5, 5), (6, 6)}, exit_location=[17, 14], num_people=Total_People)
    print("环境初始化完成")
    
    print("初始化DQN Agent...")
    state_size = env.reset().shape[0]
    action_size = 5  # Up, Down, Left, Right, Stay
    agent = DQNAgent(state_size, action_size)
    print(f"Agent初始化完成，状态空间大小: {state_size}, 动作空间大小: {action_size}")
    
    episodes = 500  # 减少训练轮数
    print(f"开始训练，总轮数: {episodes}")
    reward_history = []

    try:
        for e in range(episodes):
            print(f"\n开始第 {e+1} 轮训练")
            state = env.reset()
            total_reward = 0
            for step in range(500):  # 减少最大步数
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                # 每10步更新一次可视化
                if step % 10 == 0:
                    env.visualize(reward_history)
                
                if done:
                    print(f"Episode {e + 1}/{episodes}, Total Reward: {total_reward}, Steps: {step}, Time: {step * 0.001:.3f}s")
                    break
            reward_history.append(total_reward)
            if len(agent.memory) > 32:
                agent.replay(32)
    except Exception as ex:
        print(f"训练过程中发生错误: {ex}")
        import traceback
        traceback.print_exc()
    finally:
        print("训练结束，保存结果...")
        plt.ioff()  # 关闭交互模式
        plt.savefig('training_result.png')
        print("结果已保存到 training_result.png")