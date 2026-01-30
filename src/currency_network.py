import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import *

from src.abstract_node import Node, AgeNode


class Network():
    def __init__(self, nodes: List[Node], skip_null_objects=False):
        self.skip_null_objects = skip_null_objects
        ordered_nodes = []
        unordered_nodes = set(nodes)
        while unordered_nodes:
            for node in unordered_nodes.copy():
                if not node.parents:
                    ordered_nodes.insert(0, node)
                    unordered_nodes.remove(node)
                elif all(parent in ordered_nodes for parent in node.parents):
                    ordered_nodes.append(node)
                    unordered_nodes.remove(node)
        self.nodes: List[Node] = ordered_nodes

    def __str__(self):
        return f"Network(name={self.name}, nodes={self.nodes})"
    
    def list_nodes(self):
        print(f"Nodes in network:")
        for index, node in enumerate(self.nodes):
            print(f"{index}: {node} (with parents: {[parent.attribute for parent in node.parents]})")


    def estimate(self, data: pd.DataFrame, time_column: str, time_unit = 'integer'):
        time_steps = data[time_column].unique()
        current_time = min(time_steps)
        end_time = max(time_steps)

        #create a dictionary with one item per age node as keys and a list of currency estimates as values
        currency_map = {f'{node.attribute}_currency': {} for node in self.nodes if isinstance(node, AgeNode)}

        
        while current_time <= end_time:
            if current_time in time_steps:
                current = data[data[time_column] == current_time]

            #update model nodes, respecting dependency order
            for node in self.nodes:
                node.update_state(current)
                node.update_belief()

            # put the currency estimates in the currency_map
            if not self.skip_null_objects or current_time in time_steps:
                for n in self.nodes:
                    if isinstance(n, AgeNode):
                        currency_map[f'{n.attribute}_currency'][current_time] = n.currency()
            
            current_time = current_time + 1

        return currency_map
    

    def plot_network(self):
        pass

    def clear(self):
        for node in self.nodes:
            node.clear()   