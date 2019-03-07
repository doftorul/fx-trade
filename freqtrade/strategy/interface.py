"""
Strategy interface
This module defines the interface to apply for strategies
"""

from freqtrade.strategy import *
import importlib

def retrieve_strategy(name_strategy):
    module_name, class_name = "freqtrade.strategy", name_strategy
    clazz = getattr(
        importlib.import_module(module_name), 
        class_name
        )
    return clazz
