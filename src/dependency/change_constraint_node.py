import numpy as np
import pandas as pd
from typing import *

from src.abstract_node import *


def uniform_range_decay(current_timestep, interval_range):
    if current_timestep < 0 or current_timestep > interval_range:
        return 0.0
    else:
        return 1/(interval_range+1)


class ChangeConstraint(AgeNode):
    def __init__(self, attribute, parent: AgeNode, range: tuple[int, int], decay:Callable[[int], float] = uniform_range_decay, certainty_on_reobservation: bool = False): # left and right range? no integrated interval function in python but libraries exist
        super().__init__(attribute, certainty_on_reobservation)
        self.type = 'Change Constraint Node'
        self.parents = {parent}
        self.dependency = parent
        self.range = range
        self.decay = decay # function that gives the change probability; could simply be a uniform distribution over the range as long as the current age is within the range, and 0 otherwise
        self.registry: list[Event] = []
        self.last_event: Event | None = None

    def __str__(self):
        return self.type

    def get_last_event(self):
        return self.last_event

    def update_state(self, current):
        super().update_state(current)
        
        # PARENT re-observation
        if self.dependency.get_last_event().get_special_event_type() is not None:
            self.registry = []
        

        # reset after new attribute value
        if self.age == 0:
            self.registry = []
            self.last_event = Event.pointmass(0, 1.0)

        # re-observation of the same attribute value
        elif self.age > 0 and self.identical_observation_trigger:
            parent_event = self.dependency.get_last_event()
            self.registry = [event for event in self.registry if (event.min_t() < (max(self.range)))]
            for event in self.registry:
                event.grow()
            self.registry.insert(0, parent_event)
            
            self.last_event = Event.reset(self.age)

        # regular update: no value at current timestep
        else:
            parent_event = self.dependency.get_last_event()

            ## UPDATE REGISTRY
            # prune events from registry that have no effect on the belief anymore
            max_age_with_belief = max(age for age, prob in self.age_map.items() if prob > 1e-15)
            self.registry = [event for event in self.registry if (event.min_t() < (max_age_with_belief + max(self.range)))]
            
            # grow the events in the registry
            for event in self.registry:
                event.grow()

            self.registry.insert(0, parent_event)
            
            ## UPDATE LAST EVENT
            # convolve
            parent_times = parent_event.event_times
            child_times = {}
            width = max(self.range) - min(self.range)
            for parent_time, parent_prob in parent_times.items():
                for i in range(min(self.range), max(self.range)+1):
                    child_times[parent_time - i] = child_times.get(parent_time - i, 0.0) + (parent_prob * self.decay(i - min(self.range), width))

            self.last_event = Event(child_times)



    def _update_belief_uncertainty(self):
        # survival function S(k) for k = 0, 1, ..., age+1 ;  S(k) = probability that no registry entry induced a child age in {0, ..., k-1}
        survival = [1.0 for i in range(self.age+1)] + [0.0]
        # if this node had a reobservation, the survival function should only contain new ages after the reobservation
        if self.age_of_certain_remeasurement is not None:
            cutoff_reobservation = (self.age) - (self.age_of_certain_remeasurement)
            for k in range(1, cutoff_reobservation+1):
                for event in self.registry:
                    survival[k] *= (1 - self.infer(event, k))
            survival[self.age+1] = (1 - survival[cutoff_reobservation])

        # regular case, all registry entries can induce changes in the child node
        else:
            for k in range(1,self.age+2):
                for event in self.registry:
                    survival[k] *= (1 - self.infer(event, k))

        # convert to belief distribution / age_map
        new_belief = AgeMap()
        for k in range(self.age+1):
            new_belief[k] = survival[k] - survival[k+1]
        self.age_map = new_belief.copy()


    def infer(self, event: Event, k: int) -> float:
        # extra check to avoid an immediate dip in the survival function
        if k <= 0:
            return 0.0
        
        # infer how much probability mass from the parent event would have induced a change in the child node less than k timesteps ago
        width = max(self.range) - min(self.range)
        mass = 0.0
        for parent_time, parent_prob in event.event_times.items():
            for i in range(min(self.range), max(self.range)+1):
                child_time = parent_time - i
                if 0 <= child_time < k:
                    mass += parent_prob * self.decay(i - min(self.range), width)
        return mass
    

    def clear(self) -> None:
        super().clear()
        self.registry = []
        self.last_event = None