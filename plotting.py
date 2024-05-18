import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import pickle
import copy
import lens
from gpt_2_dataset import GPT2Dataset
import argparse
from lens import Logit_lens, Linear_lens
from lens_model import Lens_model
from train import validate
import matplotlib.pyplot as plt

def evaluate(lens_model, model_name, dataset_path, max_length=64, device=torch.device('cpu'), output_path=None, batch_size=2, loss_function_name = "kl"):
    with open(dataset_path, "rb") as file:
        dataset = pickle.load(file)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    dataset.set_device(device)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)    

    if loss_function_name == "kl":
        loss_function = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

    elif loss_function_name == "mse":
        loss_function = torch.nn.MSELoss()
    
    elif loss_function_name == "ce":
        loss_function = torch.nn.CrossEntropyLoss()

    lens_model.to(device)
    lens_model.eval()

    losses = np.array([0.0 for _ in range(len(lens_model.layers))])
    probabilities = [[] for _ in range(len(lens_model.layers))]
    ranks = [[] for _ in range(len(lens_model.layers))]

    for i, batch in enumerate(loader):
        data, input_ids, targets, attention_mask, target_index = batch
        logits = lens_model.forward(input_ids, attention_mask, targets)
        lens_probs = lens_model.get_correct_class_probs(input_ids, attention_mask, targets, target_index)
        model_logits = model(input_ids, attention_mask=attention_mask).logits
        model_logits = model_logits[torch.arange(model_logits.shape[0]), target_index-1]

        batch_loss = np.array([0.0 for _ in range(len(lens_model.layers))])

        for layer in lens_model.layers:
            #breakpoint()
            layer_logits = logits[torch.arange(logits.shape[0]), target_index-1, :,[(layer - 1) for _ in range(logits.shape[0])]]
            c_ranks = torch.argsort(layer_logits, descending=True).cpu().detach().numpy()
            for b in range(logits.shape[0]):
                rank = np.where(c_ranks[b] == targets[b][target_index[b]].cpu().detach().numpy())[0]
                ranks[layer-1].append(rank)
                probabilities[layer-1].append(lens_probs[b, layer-1].cpu().detach().numpy())

            if loss_function_name == "kl" or loss_function_name == "mse":
                log_probs_model = torch.nn.functional.log_softmax(model_logits, dim=-1)
                log_probs_lens = torch.nn.functional.log_softmax(layer_logits, dim=-1)
                batch_size_t = target_index.shape[0]
                loss = loss_function(log_probs_lens, log_probs_model)
                batch_loss[layer-1] = loss.cpu().detach().numpy()  

            elif loss_function_name == "ce":
                ##TODO: Implement this
                raise NotImplementedError
        breakpoint()   
        losses += batch_loss

        if i % int(len(loader)*0.1) == 0:
            print(f"    Batch: {i}/{len(loader)}, Loss: {loss.item()}")

    probabilities = np.array(probabilities)
    ranks = np.array(ranks)        
    average_loss = losses / len(dataset)
    average_probability = np.mean(probabilities, axis=-1)
    average_rank = np.mean(ranks, axis=1)

    if output_path is not None:
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))
        axes[0].plot(average_loss)
        axes[0].set_title("Average loss")
        axes[1].plot(average_probability)
        axes[1].set_title("Average probability")
        axes[2].plot(average_rank)
        axes[2].set_title("Average rank")
        plt.savefig(output_path)
    

def main(lens_model_path, data_path="./datasets/dataset_train.pkl", device=torch.device('cpu'), batch_size=1):
    lens_model = torch.load(lens_model_path)
    lens_model.to(device)
    lens_model.eval()
    
    losses, average_loss = validate(lens_model, data_path, device, batch_size)

        
if __name__ == "__main__":
    with open("models/lens_model_cpu.pkl", "rb") as file:
        lens_model = pickle.load(file)

    evaluate(lens_model, "gpt2", "./datasets/dataset_test.pkl", device=torch.device('cpu'), batch_size=2, loss_function_name="kl")