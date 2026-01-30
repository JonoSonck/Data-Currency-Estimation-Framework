import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import *



''' Abstract class for a node in a belief network '''
class Node(ABC):
    
    # class variable and method to add an incremetal id to the nodes
    id_counter = 0
    @classmethod
    def set_id(cls):
        cls.id_counter += 1
        return cls.id_counter

    def __init__(self, attribute, parents=set()):
        self.attribute = attribute
        self.id = self.set_id()
        self.parents: Set[Node] = set(parents)
        self.type = "Node"

    def get_attribute(self):
        return self.attribute
    
    def get_id(self):
        return self.id


    def get_type(self):
        return self.type

    def get_parents(self):
        return self.parents

    # Updates the node state based on new evidence
    @abstractmethod
    def update_state(self, current):
        pass

    # Updates the node belief
    @abstractmethod
    def update_belief(self):
        pass

    # Clears the nodes
    @abstractmethod
    def clear(self):
        pass



'''Class for an age node in a belief network, which inherets from the abstract Node class'''
class AgeNode(Node):
    def __init__(self, attribute):
        super().__init__(attribute)
        self.type = 'Age Node'
        self.previous_value = None
        self.age = None
        self.age_map = {}

    def __str__(self):
        return f"Age node for attribute '{self.attribute}'"

    def belief(self):
        return self.age_map
    
    def probability(self, event):
        return self.age_map[event]

    def currency(self):
        return self.probability(self.age)

    # update the state of a node with new record
    def update_state(self, current):
        if current.empty:
            if self.age == None:
                self.age = 0
            else:
                self.age += 1
        
        c_value = current[self.attribute].iloc[0]
        if pd.notna(c_value) and c_value != self._get_previous_value():
            self._set_age(0)
            self._set_previous_value(c_value)

        else:
            if self.age == None:
                self.age = 0
            else:
                self.age += 1


    def update_belief(self):
        pass

    def _set_previous_value(self, value): 
        self.previous_value = value

    def _get_previous_value(self):
        return self.previous_value
    

    def _set_age(self, age):
        self.age = age

    def _get_age(self):
        return self.age

    def _get_age_map(self):
        return self.age_map

    def clear(self):
        super().clear()
        self.age_map = {}
        self.previous_value = None
        self.age = None



'''Class for an data node in a belief network, which inherets from the abstract Node class'''
class DataNode(Node):
    def __init__(self, attribute, state=None, prior=None):
        super().__init__(attribute)
        self.type = 'Data Node'
        self.state = state
        self.prior = prior
        self.belief = None

    def __str__(self):
        return f"Data node for attribute '{self.attribute}'"
    
    def update_state(self, current):
        c_value = current[self.attribute].iloc[0]
        if pd.notna(c_value):
            self.state = c_value
    
    def update_belief(self):
        if self.state is None:
            self.belief = self.prior
        else:
            self.belief = {self.state: 1.0}
    
    def clear(self):
        super().clear()
        self.state = None
        self.belief = None