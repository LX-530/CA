# person_behavior.py
import random
import numpy as np

class Person:
    
    def __init__(self, position, map_loader):
        self.position = position
        self.escaped = False
        self.map_loader = map_loader

    def get_possible_moves(self, occupancy, dynamic_field):
        x, y = self.position
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        possible_moves = []
        min_field = np.inf

        total_field = self.map_loader.static_field + dynamic_field

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.map_loader.rows and 0 <= ny < self.map_loader.cols and
                self.map_loader.map_data[nx, ny] != 1 and occupancy[nx, ny] == 0):  # Not obstacle, not occupied
                field_value = total_field[nx, ny]
                if field_value < min_field:
                    min_field = field_value
                    possible_moves = [(nx, ny)]
                elif field_value == min_field:
                    possible_moves.append((nx, ny))

        # Include staying put if no better move, but prefer moving
        stay_field = total_field[x, y]
        if possible_moves:
            return random.choice(possible_moves)
        else:
            return self.position  # Stay if no move