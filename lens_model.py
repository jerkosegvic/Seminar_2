import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import pickle
import copy

class Lens_model(torch.nn.Module):
    def __init__(self, lens, layers, model_name="gpt2", model_path=None):
        '''
        Initializes the Lens_model class.
        model_name: str
            The name of the model to be used.
        lens: Lens
            The lens to be used.
        layers: list of int
            The layers to be used.
        '''
        super(Lens_model, self).__init__()
        
        if model_path is not None:
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
        
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.lens = lens
        self.layers = layers
        self.device = torch.device('cpu')
        self.unembed = self.model.get_output_embeddings()
        self.final_layer_norm = self.model.base_model.ln_f

    def to(self, device):
        '''
        Moves the model to the device.
        device: torch.device
            The device to be used.
        '''
        self.device = device
        self.lens.to(device)
        self.model.to(device)
        return self
        
    def forward(self, input_ids, attention_mask, targets, index):
        '''
        Forward pass of the model.
        input_ids: torch.Tensor
            The input ids.
        attention_mask: torch.Tensor
            The attention mask.
        targets: torch.Tensor
            The targets.
        index: int
            The index of the target token.
        '''
        model_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=targets, output_hidden_states=True)
        hs = torch.stack(model_outputs.hidden_states, dim = 1)
        logits