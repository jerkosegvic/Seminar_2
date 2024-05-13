import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import pickle
import copy
import abc


class Lens(abc.ABC, torch.nn.Module):
    """Abstract base class for all Lens"""
    
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.device = torch.device('cpu')
        self.output_logits = False
        self.dim_in = dim_in
        self.dim_out = dim_out

    @abc.abstractmethod
    def forward(self, h, idx):
        pass

    @staticmethod
    def from_model(model, init_parameter=None):
        rv = Lens(model.config.hidden_size, model.config.hidden_size)
        rv.dim_in = model.config.hidden_size
        rv.dim_out = model.config.hidden_size
        return rv

    
    def set_parameters(self, parameters):
        pass

    def to(self, device):
        self.device = device
    
class Logit_lens(Lens):
    def __init__(self, dim_in, dim_out):
        super().__init__(dim_in, dim_out)

    def forward(self, h, idx):
        return h[:,idx]
    
    @staticmethod
    def from_model(model, init_parameter=None):
        rv = Logit_lens(model.config.hidden_size, model.config.hidden_size)
        rv.dim_in = model.config.hidden_size
        rv.dim_out = model.config.hidden_size
        return rv
    
class Linear_lens(Lens):
    def __init__(self, dim_in, dim_out):
        super().__init__(dim_in, dim_out)
        self.linear = torch.nn.Linear(dim_in, dim_out)

    def forward(self, h, idx):
        return self.linear(h)[:,idx]
    
    def to(self, device):
        self.device = device
        self.linear.to(device)
        return self
    
    def set_parameters(self, parameters):
        self.linear.weight = torch.nn.Parameter(parameters['weight'])
        self.linear.bias = torch.nn.Parameter(parameters['bias'])

    @staticmethod
    def from_model(model, init_parameter=None):
        rv = Linear_lens(model.config.hidden_size, model.config.hidden_size)
        rv.dim_in = model.config.hidden_size
        rv.dim_out = model.config.hidden_size
        if init_parameter is not None:
            rv.set_parameters(init_parameter)

        return rv