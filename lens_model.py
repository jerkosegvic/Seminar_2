import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import pickle
import copy
import lens

class Lens_model(torch.nn.Module):
    def __init__(self, lens, layers=None, model_name="gpt2", model_path=None):
        '''
        Initializes the Lens_model class.
        model_name: str
            The name of the model to be used.
        lens: list of Lens
            The lens to be used.
        layers: list of ints
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
        self.num_layers = self.model.config.num_hidden_layers
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
    
    def get_probs(self, input_ids, attention_mask, targets, target_index):
        '''
        Gets the probabilities of the targets.
        input_ids: torch.Tensor
            The input ids.
        attention_mask: torch.Tensor
            The attention mask.
        targets: torch.Tensor
            The targets.
        '''
        output = self.forward(input_ids, attention_mask, targets)
        for i in range(len(output)):
            layer_ = self.layers[i]
            batch_size = output[i].shape[0]
            output[i] = torch.softmax(output[i][torch.arange(batch_size), target_index - 1], dim=-1)
        
        return output
    
    def get_correct_class_probs(self, input_ids, attention_mask, targets, target_index):
        '''
        Gets the probabilities of the correct .
        input_ids: torch.Tensor
            The input ids.
        attention_mask: torch.Tensor
            The attention mask.
        targets: torch.Tensor
            The targets.
        target_index: torch.Tensor
            The index of the token that we will predict
        '''
        probs = self.get_probs(input_ids, attention_mask, targets, target_index)
        batch_size = probs[0].shape[0]
        return list(map(lambda x: x[torch.arange(batch_size), targets[torch.arange(batch_size), target_index]], probs))


    def forward(self, input_ids, attention_mask, targets):
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
        output = []
        for ly, ln in zip(self.layers, self.lens):
            lens_output = ln.forward(hs, ly)
            if ln.output_logits:
                output.append(lens_output)
            
            else:
                if ly == -1 or ly == self.num_layers:
                    logits = self.unembed.forward(lens_output)
                    output.append(logits)
                else:
                    logits = self.unembed.forward(self.final_layer_norm.forward(lens_output))
                    output.append(logits)
        
        return output