import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import *
import xml.etree.ElementTree as ET

from src.abstract_node import *
from src.shelflife.basic_shelflife_node import *
from src.shelflife.conditional_shelflife_node import *
from src.shelflife.dynamic_shelflife_node import *
from src.changepoint.cusum_node import *
from src.dependency.aggregator_node import *


class Network():
    def __init__(self, nodes: List[Node], skip_null_objects: bool=False):
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

    def __str__(self) -> str:
        return f"Network(name={self.name}, nodes={self.nodes})"
    
    def list_nodes(self) -> str:
        print(f"Nodes in network:")
        for index, node in enumerate(self.nodes):
            print(f"{index}: {node} (with parents: {[parent.attribute for parent in node.parents]})")


    def estimate(self, data: pd.DataFrame, time_column: str, time_unit: str = 'integer') -> Dict[str, Dict[int, float]]:
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


    def clear(self) -> None:
        for node in self.nodes:
            node.clear()


    @classmethod
    def from_xml(cls, xml_path: str, external_context: dict = {}, skip_null_objects: bool = False):  # class_registry: dict = {}, enum_registry: dict = {},
        """
        build a network from XML config file.
        """
        class_registry = {
            "DataNode": DataNode,
            "ShelfLife": ShelfLife,
            "ConditionalShelfLife": ConditionalShelfLife,
            "DynamicShelfLife": DynamicShelfLife,
            "CUSUMPoisson": CUSUMPoisson,
            "CUSUMNormal": CUSUMNormal,
            "Aggregator": Aggregator,
        }
        enum_registry = {
            'BasicQuantifier.ALL': BasicQuantifier.AVERAGE,
            'BasicQuantifier.MOST': BasicQuantifier.MOST,
            'BasicQuantifier.MANY': BasicQuantifier.MANY,
            'BasicQuantifier.AVERAGE': BasicQuantifier.AVERAGE,
            'BasicQuantifier.SOME': BasicQuantifier.SOME,
            'BasicQuantifier.FEW': BasicQuantifier.FEW,
            'BasicQuantifier.ANY': BasicQuantifier.ANY,
        }

        tree = ET.parse(xml_path)
        root = tree.getroot()

        nodes = root.find("Nodes")
        instances = {}

        # instantiate nodes
        for node in nodes.findall("Node"):
            node_name = node.attrib["name"]
            class_name = node.attrib["class"]
            if class_name not in class_registry:
                raise Exception(f"Unknown class: {class_name}")
            cls_obj = class_registry[class_name]

            params = {}
            params["attribute"] = node.attrib["attribute"]

            # load xml parameters
            for param in node.findall("Param"):
                key = param.attrib["name"]
                value = param.text.strip()

                if param.attrib.get("type") == "int":
                    value = int(value)
                elif param.attrib.get("type") == "float":
                    value = float(value)
                elif param.attrib.get("type") == "dict":
                    import json
                    value = json.loads(value)
                # else string by default

                params[key] = value

            # add external parameters
            for key in list(params.keys()):
                if key.endswith("_ext"):
                    ext_ref_name = params.pop(key)
                    params[key.replace("_ext", "")] = external_context[ext_ref_name]

            # check for parents
            parents_elem = node.find("Parents")
            if parents_elem is not None:
                parents = []
                for parent in parents_elem.findall("Parent"):
                    parent_name = parent.attrib["ref"]
                    if parent_name not in instances:
                        raise Exception(f"Unknown parent reference: {parent_name}")
                    parents.append(instances[parent_name])
                params['parents'] = parents


            aggregator_elem = node.find('AggregatorType')
            if aggregator_elem is not None:
                params['aggregator'] = enum_registry[aggregator_elem.get('enum')]

            # instantiation method
            instantiation_method = node.findtext("Method")
            if instantiation_method:
                target = getattr(cls_obj, instantiation_method)
            else:
                target = cls_obj

            # instantiate
            obj = target(**params)
            instances[node_name] = obj
        
        # create network
        return cls(
            nodes=list(instances.values()),
            skip_null_objects=skip_null_objects
        )