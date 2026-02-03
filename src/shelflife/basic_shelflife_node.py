import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import *

from src.abstract_node import Node, AgeNode


class ShelfLife(AgeNode):
    def __init__(self, attribute: str, hazard: Callable[[int], float]):
        super().__init__(attribute)
        self.hazard = hazard
        self.type = 'Shelf Life Node'

    def __str__(self) -> str:
        return self.type

    def get_parents(self) -> Set[Node]:
        return set()
    
    def update_belief(self) -> None:
        if self.age == 0:
            self.age_map = {}
            self.age_map[0] = 1.0  
        else:
            updated_age_map = {}
            for key, value in self.age_map.items():
                updated_age_map[key + 1] = value * (1 - self.hazard.apply(self.age))
            updated_age_map[0] = self.hazard.apply(self.age)
            self.age_map = updated_age_map.copy()
        
    def get_hazard(self) -> float:
        return self.hazard.apply(self.age)
        
    def set_hazard(self, hazard: Callable[[int], float]) -> None:
        self.hazard = hazard