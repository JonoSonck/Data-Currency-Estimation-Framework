import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import *
from enum import Enum

from src.abstract_node import AgeNode


class BasicQuantifier(Enum):
    ALL     = (lambda x: 1.0 if x == 1.0 else 0.0,)
    MOST    = (lambda x: x ** 4,)
    MANY    = (lambda x: x ** 2,)
    AVERAGE = (lambda x: x,)
    SOME    = (lambda x: x ** 0.5,)
    FEW     = (lambda x: x ** 0.25,)
    ANY     = (lambda x: 0.0 if x == 0.0 else 1.0,)

    def __init__(self, func):
        self.apply = func



class QuantifiedAggregation(ABC):
    def __init__(self, quantifier):
        self.quantifier = quantifier

    def aggregate(self, values):
        pass



class ExpectedQuantity(QuantifiedAggregation):
    def __init__(self, quantifier):
        super().__init__(quantifier)
    
    def aggregate(self, values):
        n = len(values)
        output = 0.0

        #initialize arrays
        previous = [1.0] + [0.0] * n
        current = [0.0] * (n+1)

        for x in range(1, n+1):
            p = values[x-1]  # probability that a change already occurred at current timestep for parent x
            for i in range(n+1):
                current[i] = previous[i] * (1.0 - p)
                if i > 0:
                    current[i] += previous[i-1] * p
            previous = current.copy()

        for q in range(n+1):
            output += current[q] * self.quantifier.apply(q/n)

        return output
    


class Aggregator(AgeNode):
    def __init__(self, attribute, parents, aggregator = None):
        super().__init__(attribute)
        self.type = 'Quantified Aggregator Node'
        self.age = 0
        self.parents = set(parents)
        if aggregator is None:
            self.aggregator = ExpectedQuantity(BasicQuantifier.ALL)
        elif isinstance(aggregator, BasicQuantifier):
            self.aggregator = ExpectedQuantity(aggregator)
        elif isinstance(aggregator, QuantifiedAggregation):
            self.aggregator = aggregator
        else:
            print(type(aggregator), BasicQuantifier)
            raise Exception("Passed argument fo aggregator is neither a quantifier nor a quantified aggregation.")

    def __str__(self):
        return self.type

    def update_belief(self):
        if not self.age_map:
            self.age_map[0] = 1.0
            return

        updated_age_map = {0:0.0}
        for key, value in self.age_map.items():
            updated_age_map[key+1] = value

        current_cumul = 0.0
        for key, value in updated_age_map.items():
            if key < self.age:
                current_cumul += value

        next_cumul = self.aggregator.aggregate([(1.0 - parent.currency()) for parent in self.parents])

        if next_cumul > current_cumul:
            updated_age_map[0] += (next_cumul - current_cumul)
            updated_age_map[self.age] -= (next_cumul - current_cumul)
        self.age_map = updated_age_map.copy()  