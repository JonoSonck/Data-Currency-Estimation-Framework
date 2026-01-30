import numpy as np
from pandas import DataFrame
from abc import ABC, abstractmethod
from typing import *

from src.abstract_node import AgeNode, DataNode


class CrossProductIterator:
    def __init__(self, attribute_value_map):
        self.attribute_value_map = attribute_value_map
        self.mask = [(key, 0) for key in sorted(attribute_value_map.keys())]

    def has_next(self):
        return bool(self.mask)
    
    def next(self):
        if not self.mask:
            return None

        # construct the next object based on the current state of the mask
        next_object = {}
        for key, idx in self.mask:
            values = self.attribute_value_map[key]
            next_object[key] = values[idx] if values else None

        # advance over the cartesian product
        self._adapt_mask()

        return next_object

    def _adapt_mask(self):
        for i in range(len(self.mask)):
            key, index = self.mask[i]

            # as long as an element has not reached its max index, increment and return
            if index < (len(self.attribute_value_map[key]) - 1):
                self.mask[i] = (key, index+1)
                return # kicks out of the function
            
            # start again for a certain variable
            else:
                self.mask[i] = (key, 0)

        # full product made so empty the mask
        self.mask = []



''' A node that uses a Geometric prior, but the parameter is adapted based on the data.
More precisely, a co-variate variable determines which values the parameter takes through
a simple generator function. '''
class ConditionalShelfLife(AgeNode):
    def __init__(self, attribute, generator, parents):
        super().__init__(attribute)
        self.type = 'Conditional Geometric Shelf Life Node'
        self.generator = generator
        self.parents = set(parents)

    @classmethod
    def from_map(cls, attribute, map: dict, parents):
        if isinstance(map, DataFrame):
            map = map.set_index(sorted(map.columns[:-1]))[map.columns[-1]].to_dict()
        generator = lambda x: map.get(tuple(x.values()))
        return cls(attribute, generator, parents)
    

    def __str__(self):
        return self.type

    def clear(self):
        super().clear()


    def update_belief(self):
        if self.age == 0:
            self.age_map = {}
            self.age_map[0] = 1.0
        else:
            updated_age_map = {0:0.0}
            for key, value in self.age_map.items():
                updated_age_map[key+1] = value

            iterator = CrossProductIterator({parent.attribute: list(parent.belief.keys()) for parent in self.parents})

            while iterator.has_next():
                current_parent_values = iterator.next()
                    
                prior = 1.0
                for parent in self.parents:
                    parent_belief_table = parent.belief
                    prior *= parent_belief_table[current_parent_values[parent.attribute]]
                
                geom_p = self.generator(current_parent_values)  # get the hazard value from the generator function
                geom_k = self.age
                prob = geom_p * ((1 - geom_p) ** geom_k)

                updated_age_map[0] += (prior * prob)

            updated_age_map[self.age] -= updated_age_map[0]

            self.age_map = updated_age_map.copy()
