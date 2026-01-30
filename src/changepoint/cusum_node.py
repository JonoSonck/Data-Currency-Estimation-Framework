import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import *

from src.abstract_node import AgeNode


class CUSUM(AgeNode, ABC):
    def __init__(self, attribute, ground_belief=1.0, cusum=0, a=1, b=1):
        super().__init__(attribute)
        self.type = 'AbstractCUSUM Node'
        self.ground_belief = ground_belief
        self.cusum = cusum
        self.a = a
        self.b = b
    
    def __str__(self):
        return f"CUSUM node for attribute '{self.attribute}'"
    
    def cusum_belief(self):
        return {'ground':self.ground_belief, 'alternative':(1-self.ground_belief)}
    
    def update_belief(self):
        # update the cusum belief
        alt_prob = np.tanh((self.cusum**self.a)/self.b)
        self.ground_belief = 1-alt_prob

        # update the age_map belief
        if not self.age_map:
            self.age_map[0] = 1.0
            return

        updated_age_map = {0:0.0}
        for key, value in self.age_map.items():
            updated_age_map[key+1] = value

        prev_cumul = 0.0

        for key, value in updated_age_map.items():
            if key < self.age:
                prev_cumul += value

        new_cumul = alt_prob

        if new_cumul > prev_cumul:
            updated_age_map[0] += (new_cumul - prev_cumul)
            updated_age_map[self.age] -= (new_cumul - prev_cumul)

        self.age_map = updated_age_map.copy()
    
    def set_cusum(self, cusum):
        self.cusum = cusum

    def clear(self):
        super().clear()
        self.cusum = 0.0



class CUSUMPoisson(CUSUM):
    def __init__(self, attribute, a=1, b=1, ground_lambda=1.0, alt_lambda=2.0):
        super().__init__(attribute, a=a, b=b)
        self.type = 'CUSUM Poisson Node'
        if ground_lambda <= 0 or alt_lambda <= 0:
            raise Exception("Lambda values must be positive.")
        self.delta_time = 1
        self.ground_lambda = ground_lambda
        self.alt_lambda = alt_lambda

    def __str__(self):
        return f"CUSUM Poisson node for attribute '{self.attribute}'"
      
    def update_state(self, current):
        if self.age == None:
            self.age = 0
        else:
            self.age += 1

        if (current.empty) | (current[str(self.attribute)].iloc[0] is None):
            self.delta_time += 1
        else:
            k = current[str(self.attribute)].iloc[0]

            loglikelihoodratio = k * np.log(self.delta_time * self.alt_lambda)  +  self.delta_time * self.ground_lambda  -  k * np.log(self.delta_time * self.ground_lambda)  -  self.delta_time * self.alt_lambda
            self.set_cusum(max(0, (self.cusum + loglikelihoodratio)))

            self.delta_time = 1
    
    def clear(self):
        super().clear()
        self.delta_time = 1



class CUSUMNormal(CUSUM):
    def __init__(self, attribute, a=1, b=1, ground_avg=0.0, ground_std=1.0, alt_avg=2.0, alt_std=1.0):
        super().__init__(attribute, a=a, b=b)
        self.type = 'CUSUM Normal Node'
        if ground_std <= 0 or alt_std <= 0:
            raise Exception("Standard deviation values must be positive.")
        self.delta_time = 1
        self.ground_avg = ground_avg
        self.ground_std = ground_std
        self.alt_avg = alt_avg
        self.alt_std = alt_std

    def __str__(self):
        return f"CUSUM Normal node for attribute '{self.attribute}'"
    
    def update_state(self, current):
        if (current.empty) | (current[str(self.attribute)].iloc[0] is None):
            self.delta_time += 1
        else:
            k = current[str(self.attribute)].iloc[0]
            
            loglikelihoodratio_unequal_variance = 1/2 * np.log((self.ground_std**2) / (self.alt_std**2)) + (((k - self.ground_avg)**2) / (2*(self.ground_std**2))) - (((k - self.alt_avg)**2) / (2*(self.alt_std**2)))

            self.set_cusum(max(0.0, self.cusum + loglikelihoodratio_unequal_variance))

            self.delta_time = 1
    
    def clear(self):
        super().clear()
        self.delta_time = 1