# fire_model.py
import numpy as np

class FireModel:
    def __init__(self, fire_positions, rows, cols):
        self.fire_positions = fire_positions
        self.rows = rows
        self.cols = cols
        # Constants for repulsion field
        self.A = 10  # Amplitude
        self.B = 0.5    # Decay rate
        self.cutoff_radius = 4.0  # 设置一个影响半径

    def compute_dynamic_field(self):
        dynamic_field = np.zeros((self.rows, self.cols))
        for i in range(self.rows):
            for j in range(self.cols):
                for fx, fy in self.fire_positions:
                    dist = np.sqrt((i - fx)**2 + (j - fy)**2)
                    if 0 < dist <= self.cutoff_radius:
                        dynamic_field[i, j] += self.A * np.exp(-self.B * dist)
                    elif dist== 0:
                        dynamic_field[i, j] = np.inf  # Fire position itself
        return dynamic_field

    # For future: add fire spread method
    def spread_fire(self, map_data):
        pass  # Implement later if needed