# main.py
import numpy as np
import matplotlib.pyplot as plt
from map_loader import MapLoader
from person_behavior import Person
from fire_model import FireModel
import random

def visualize(map_data, persons, fires, step):
    vis_map = np.copy(map_data)
    for person in persons:
        x, y = person.position
        vis_map[x, y] = 4  # Person marker
    for fx, fy in fires:
        vis_map[fx, fy] = 3  # Fire

    plt.imshow(vis_map, cmap='hot', interpolation='nearest')
    plt.title(f'Step {step}')
    plt.show(block=False)
    plt.pause(0.05)
    plt.clf()

# Main simulation
map_loader = MapLoader('map.json')
#目标放人的区域 格式为：(start_row, end_row, start_col, end_col)
target_area = (3, 32, 2, 7) 

# 找出目标区域内所有空的单元格
target_empty_positions = []
for r in range(target_area[0], target_area[1]):
    for c in range(target_area[2], target_area[3]):
        if map_loader.map_data[r, c] == 0:
            target_empty_positions.append((r, c))
# 定义人数
num_persons = 150 
persons = []

# 从特定区域的空单元格中随机选择位置
for _ in range(min(num_persons, len(target_empty_positions))):
    pos = random.choice(target_empty_positions)
    target_empty_positions.remove(pos)
    persons.append(Person(pos, map_loader))

fire_model = FireModel(map_loader.fires, map_loader.rows, map_loader.cols)

# Simulation loop
max_steps = 300
for step in range(max_steps):
    dynamic_field = fire_model.compute_dynamic_field()

    # Occupancy for next step
    next_occupancy = np.zeros((map_loader.rows, map_loader.cols), dtype=int)
    next_positions = {}

    for person in persons:
        if not person.escaped:
            next_pos = person.get_possible_moves(next_occupancy, dynamic_field)
            if next_pos in next_positions:
                # Conflict: randomly assign or stay, simple: one stays
                if random.random() > 0.5:
                    next_positions[next_pos] = person
                # Else stay
            else:
                next_positions[next_pos] = person
                next_occupancy[next_pos[0], next_pos[1]] = 1

    # Update positions
    escaped = []
    for person in persons:
        if person in next_positions.values():
            person.position = [pos for pos, p in next_positions.items() if p == person][0]
            if person.position in map_loader.exits:
                person.escaped = True
                escaped.append(person)
        # Else stayed due to conflict

    # Remove escaped
    for p in escaped:
        persons.remove(p)

    # Visualize
    visualize(map_loader.map_data, persons, map_loader.fires, step)

    if not persons:
        print("All persons escaped!")
        break

plt.close()
