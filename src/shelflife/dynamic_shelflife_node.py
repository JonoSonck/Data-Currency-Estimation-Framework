import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import *

from src.shelflife.basic_shelflife_node import ShelfLife


class DynamicShelfLife(ShelfLife):
    def __init__(self, attribute, p, smoothing_factor):
        super().__init__(attribute, p)
        self.type = 'Dynamic Geometric Shelf Life Node'
        self.initial_p = p
        self.smoothing_factor = smoothing_factor

    def __str__(self):
        return self.type
    
    # update the state of a node with new record
    def update_state(self, current):
        c_value = current[self.attribute].iloc[0]
        if pd.notna(c_value) and c_value != self._get_previous_value():
            if self._get_previous_value() != None:
                estimate = self.smoothing_factor * self.age  +  (1 - self.smoothing_factor) * (1/self.p)
                self.set_p(1 / estimate)
        
        super().update_state(current)

    
    def clear(self):
        super().clear()
        self.set_p(self.initial_p)