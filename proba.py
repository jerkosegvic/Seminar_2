from gpt_2_dataset import GPT2Dataset
from gpt_2_dataset import GPT2Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pickle
import argparse
from lens import Logit_lens, Linear_lens
from lens_model import Lens_model
from torch.utils.data import DataLoader

model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name)
lens = []
layers = []
for layer in range(12):
    nl = Linear_lens.from_model(model)
    nl.set_parameters({'weight': torch.nn.Parameter(torch.eye(model.config.hidden_size)),\
                        'bias': torch.nn.Parameter(torch.zeros(model.config.hidden_size))})
    lens.append(nl)
    layers.append(layer)

lens_model = Lens_model(lens, layers, model_name=model_name)

with open("datasets/dataset_test.pkl", "rb") as f:
    dataset_test = pickle.load(f)

with open("datasets/dataset_train.pkl", "rb") as f:
    dataset_train = pickle.load(f)

data_train = DataLoader(dataset_train, batch_size=4, shuffle=False)
data_test = DataLoader(dataset_test, batch_size=4, shuffle=False)

data, input_ids, target_ids, attention_mask, target_index = next(iter(data_train))

lens_model.train()
print(lens_model.forward(input_ids, attention_mask, target_ids))