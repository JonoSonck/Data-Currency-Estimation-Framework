import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import *

from src.abstract_node import Node, AgeNode, AgeMap

class weibull_hazard:
    def __init__(self, p = 1.0, beta = 1.0):
        self.p = p
        self.beta = beta

    def apply(self, t: int):
        return 1.0 - (1.0 - self.p)**((t+1)**self.beta - t**self.beta)
    
class fisk_hazard:
    def __init__(self, alpha = 1.0, beta = 1.0):
        self.alpha = alpha
        self.beta = beta

    def apply(self, t: int):
        return ((self.beta / self.alpha) * ((t / self.alpha)**(self.beta - 1))) / (1 + (t / self.alpha)**self.beta)


class ShelfLife(AgeNode):
    def __init__(self, attribute: str, hazard: Callable[[int], float] = weibull_hazard(), certainty_on_reobservation: bool = False):
        super().__init__(attribute, certainty_on_reobservation)
        self.hazard = hazard
        self.type = 'Shelf Life Node'

    def __str__(self) -> str:
        return self.type

    def get_parents(self) -> Set[Node]:
        return set()
    
    def _update_belief_uncertainty(self) -> None:
        updated_age_map = {}
        for key, value in self.age_map.items():
            updated_age_map[key + 1] = value * (1 - self.hazard.apply(self.age))
        updated_age_map[0] = self.hazard.apply(self.age)
        self.age_map = AgeMap(updated_age_map)
        
    def get_hazard(self) -> float:
        return self.hazard.apply(self.age)
        
    def set_hazard(self, hazard: Callable[[int], float]) -> None:
        self.hazard = hazard