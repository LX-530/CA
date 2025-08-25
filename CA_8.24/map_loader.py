import json
import numpy as np
from collections import deque

class MapLoader:
    def __init__(self, file_path):
        self.map_data = self.load_map(file_path)
        self.rows, self.cols = self.map_data.shape
        self.exits = self.find_positions(2)
        self.fires = self.find_positions(3)
        self.static_field = self.compute_static_field()

    def load_map(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return np.array(data)

    def find_positions(self, value):
        positions = []
        for i in range(self.rows):
            for j in range(self.cols):
                if self.map_data[i, j] == value:
                    positions.append((i, j))
        return positions

    def compute_static_field(self):
        static_field = np.full((self.rows, self.cols), np.inf)
        for i in range(self.rows):
            for j in range(self.cols):
                if self.map_data[i, j] == 1:  # Obstacle
                    static_field[i, j] = np.inf
                elif self.map_data[i, j] == 2:  # Exit
                    static_field[i, j] = 0

        queue = deque(self.exits)
        visited = np.zeros((self.rows, self.cols), dtype=bool)
        for exit_pos in self.exits:
            static_field[exit_pos] = 0
            visited[exit_pos] = True

        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        while queue:
            x, y = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.rows and 0 <= ny < self.cols and not visited[nx, ny] and self.map_data[nx, ny] != 1:
                    static_field[nx, ny] = min(static_field[nx, ny], static_field[x, y] + 1)
                    visited[nx, ny] = True
                    queue.append((nx, ny))

        return static_field
