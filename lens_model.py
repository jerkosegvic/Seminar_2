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
        assert len(lens) == len(layers)
        self.lens = torch.nn.ParameterList(lens)
        self.layers = layers
        self.device = torch.device('cpu')
        self.num_layers = self.model.config.num_hidden_layers
        self.unembed = self.model.get_output_embeddings()
        self.final_layer_norm = self.model.base_model.ln_f
        self.unembed.requires_grad = False
        self.final_layer_norm.requires_grad = False

    def to(self, device):
        '''
        Moves the model to the device.
        device: torch.device
            The device to be used.
        '''
        self.device = device
        for l in self.lens:
            l.to(device)

        self.model.to(device)
        self.unembed.to(device)
        self.final_layer_norm.to(device)
    
    def get_probs(self, input_ids, attention_mask, targets, target_index):
        '''
        Gets the probabilities of the targets.
        input_ids: torch.Tensor
            The input ids.
        attention_mask: torch.Tensor
            The attention mask.
        targets: torch.Tensor
            The targets.
        target_index: torch.Tensor
            The index of the token that we will predict
        Output: torch.Tensor
            The probabilities of the targets. The shape is (batch_size, vocab_size, num_layers)
        '''
        output = self.forward(input_ids, attention_mask, targets)
        '''
        for i in range(len(output)):
            layer_ = self.layers[i]
            batch_size = output[i].shape[0]
            output[i] = torch.softmax(output[i][torch.arange(batch_size), target_index - 1], dim=-1)
        '''
        logits = output[torch.arange(output.shape[0]), target_index-1]
        probs = torch.softmax(logits, dim=-2)
        return probs
    
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
        Output: torch.Tensor
            The probabilities of the correct class. The shape is (batch_size, num_layers)
        '''
        probs = self.get_probs(input_ids, attention_mask, targets, target_index)
        return probs[torch.arange(probs.shape[0]), targets[torch.arange(targets.shape[0]), target_index]]


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
        Output: torch.Tensor
            The output of the model. The shape is (batch_size, max_length, vocab_size, num_layers)
        '''
        
        self.model.eval()
        with torch.no_grad():
            model_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=targets, output_hidden_states=True)

        hs = torch.stack(model_outputs.hidden_states, dim = 1)
        #breakpoint()
        output = []
        for ly, ln in zip(self.layers, self.lens):
            lens_output = ln.forward(hs, ly)
            if ln.output_logits:
                output.append(lens_output)
            
            else:
                if ly == -1 or ly == self.num_layers:
                    #with torch.no_grad():
                    logits = self.unembed.forward(lens_output)
                    
                    output.append(logits)
                else:
                    #with torch.no_grad():
                    logits = self.unembed.forward(self.final_layer_norm.forward(lens_output))
                    
                    output.append(logits)
        
        return torch.stack(output, dim=-1)