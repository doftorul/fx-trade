"""
Strategy interface
This module defines the interface to apply for strategies
"""
import importlib

def retrieve_strategy(name_strategy):
    clazz = getattr(
        importlib.import_module("freqtrade.strategy"), 
        name_strategy
        )
    return clazz

class Instrument:
    def __init__(self, name, time):
        self.name = name
        self.time = time
