import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import *


class AgeMap(dict):
    THRESHOLD = 1e-10

    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, 0.0 if value < self.THRESHOLD else value)

    def update(self, *args, **kwargs) -> None:
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def copy(self):
        return AgeMap(self)



''' Abstract class for a node in a belief network '''
class Node(ABC):
    
    # class variable and method to add an incremetal id to the nodes
    id_counter = 0
    @classmethod
    def set_id(cls) -> int:
        cls.id_counter += 1
        return cls.id_counter

    def __init__(self, attribute: str, parents: Optional[List[Self]] = set()):
        self.attribute = attribute
        self.id = self.set_id()
        self.parents: Set[Self] = set(parents)
        self.type = "Node"

    def get_attribute(self) -> str:
        return self.attribute
    
    def get_id(self) -> int:
        return self.id


    def get_type(self) -> str:
        return self.type

    def get_parents(self) -> Set[Self]:
        return self.parents

    # Updates the node state based on new evidence
    @abstractmethod
    def update_state(self, current: pd.DataFrame) -> None:
        pass

    # Updates the node belief
    @abstractmethod
    def update_belief(self) -> None:
        pass

    # specific update_belief segment for cases where no value was observed at timestep
    @abstractmethod
    def _update_belief_uncertainty(self) -> None:
        pass

    # Clears the nodes
    @abstractmethod
    def clear(self) -> None:
        pass



'''Class for an age node in a belief network, which inherets from the abstract Node class'''
class AgeNode(Node):
    def __init__(self, attribute: str, certainty_on_reobservation: bool = False):
        super().__init__(attribute)
        self.type = 'Age Node'
        self.previous_value = None
        self.age = None
        self.age_map = AgeMap()
        self.certainty_on_reobservation = certainty_on_reobservation
        self.identical_observation_trigger = False

    def __str__(self) -> str:
        return f"Age node for attribute '{self.attribute}'"

    def belief(self) -> Dict:
        return self.age_map
    
    def probability(self, event: int) -> float:
        return self.age_map[event]

    def currency(self) -> float:
        return self.probability(self.age)

    # update the state of a node with new record
    def update_state(self, current: pd.DataFrame) -> None:
        if current.empty:
            if self.age == None:
                self.age = 0
            else:
                self.age += 1
        
        else:
            c_value = current[self.attribute].iloc[0]
            # observation with same value 
            if pd.notna(c_value) and c_value == self._get_previous_value() and self.certainty_on_reobservation:
                self.identical_observation_trigger = True
            
            # observation with different value
            if pd.notna(c_value) and c_value != self._get_previous_value():
                self._set_age(0)
                self._set_previous_value(c_value)

            else:
                if self.age == None:
                    self.age = 0
                else:
                    self.age += 1

            

        

    def update_belief(self) -> None:
        # if a new observation is different from the last known value
        if self.age == 0:
            self.age_map = AgeMap()
            self.age_map[0] = 1.0
            return
        
        # if we measure it again, but the value stays the same, we can be certain that the age is still the same and previous intermediary probabilities drop to 0
        if self.identical_observation_trigger:
            self.age_map = AgeMap({i: 0.0 if i < self.age else 1.0 for i in range(self.age + 1)})
            self.identical_observation_trigger = False
            return
        
        # child classes handle behaviour for cases where there is no attribute value observed at the current timestep
        self._update_belief_uncertainty()
    

    def _update_belief_uncertainty(self) -> None:
        pass

    def _set_previous_value(self, value: Optional[Any]) -> None: 
        self.previous_value = value

    def _get_previous_value(self) -> Optional[Any]:
        return self.previous_value
    

    def _set_age(self, age: Optional[int]) -> None:
        self.age = age

    def _get_age(self) -> Optional[int]:
        return self.age

    def _get_age_map(self) -> Dict:
        return self.age_map

    def clear(self) -> None:
        super().clear()
        self.age_map = AgeMap()
        self.previous_value = None
        self.age = None



'''Class for an data node in a belief network, which inherets from the abstract Node class'''
class DataNode(Node):
    def __init__(self, attribute: str, state: Optional[Any]=None, prior: Optional[Dict]=None):
        super().__init__(attribute)
        self.type = 'Data Node'
        self.state = state
        self.prior = prior
        self.belief = None

    def __str__(self) -> str:
        return f"Data node for attribute '{self.attribute}'"
    
    def update_state(self, current: pd.DataFrame) -> None:
        c_value = current[self.attribute].iloc[0]
        if pd.notna(c_value):
            self.state = c_value
    
    def update_belief(self) -> None:
        if self.state is None:
            self.belief = self.prior
        else:
            self.belief = {self.state: 1.0}
    
    def _update_belief_uncertainty(self) -> None:
        pass
    
    def clear(self) -> None:
        super().clear()
        self.state = None
        self.belief = None