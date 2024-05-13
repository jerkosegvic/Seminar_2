from gpt_2_dataset import GPT2Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pickle
import argparse
from lens import Logit_lens, Linear_lens
from lens_model import Lens_model

def validate(lens_model, model, dataset_path, max_length=64, device=torch.device('cpu'), output_path=None, batch_size=2, loss_function_name = "kl"):
    '''
    -len_model: Lens_model
        The lens model to be used.
    -model_name: str
        The name of the model to be used.
    -max_length: int
        The maximum length of the input sequence in tokens.
    -dataset_path: str
        The path to the dataset.
    -device: string
        The name of device which is going to be used.
    -output_path: str
        The path to the output file.
    '''
    valid_len = 100
    layers = lens_model.layers

    with open(dataset_path, "rb") as file:
        dataset = pickle.load(file)

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
    model.eval()

    losses = []
    print("Validating...")
    for i, batch in enumerate(loader):
        data, input_ids, targets, attention_mask, target_index = batch
        if loss_function_name == "kl" or loss_function_name == "mse":
            base_model_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            base_model_logits = torch.stack([base_model_logits for layer in layers], dim=-1)
            output = lens_model.forward(input_ids, attention_mask, targets)
            log_probs_model = torch.nn.functional.log_softmax(base_model_logits, dim=-2)
            log_probs_lens = torch.nn.functional.log_softmax(output, dim=-2)
            batch_size_t = target_index.shape[0]
            loss = loss_function(log_probs_lens[torch.arange(batch_size_t), target_index], log_probs_model[torch.arange(batch_size_t), target_index])

        if loss_function_name == "ce":
            output = lens_model.get_probs(input_ids, attention_mask, targets, target_index)
            batch_size_t = target_index.shape[0]
            targets_ = targets[torch.arange(batch_size_t), target_index]
            targets_ = torch.stack([targets_ for layer in layers], dim=1)
            loss = loss_function(output, targets_)

        losses.append(loss.item())
        if i % int(len(loader)*0.1) == 0:
            print(f"    Batch: {i}/{len(loader)}, Loss: {loss.item()}")

        if i == valid_len:
            break

    average_loss = sum(losses) / len(losses)
    print(f"Average loss: {average_loss}")
    return losses, average_loss



def train(model_name="gpt2", max_length=64, dataset_train_path=None, dataset_test_path=None, \
         device=torch.device('cpu'), output_path=None, layers=[-1], loss_function_name="kl",\
         num_epochs=10, batch_size=2, lr=0.001, save_interval=1):
    '''
    -model_name: str
        The name of the model or the path to the model to be used.
    -dataset_train_path: str
        The path to the training dataset.
    -dataset_test_path: str
        The path to the testing dataset.
    -device: torch.device
        The device to be used.
    -output_path: str
        The path to the output file.
    -layers: list of ints
        The layers to be used.
    -loss_function_name: str
        The name of the loss function to be used.
    -num_epochs: int
        The number of epochs.
    -batch_size: int
        The batch size.
    -lr: float
        The learning rate.
    -save_interval: int
        The interval of saving the model.
    '''
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    lens = []
    for layer in layers:
        nl = Linear_lens.from_model(model)
        nl.set_parameters({'weight': torch.nn.Parameter(torch.eye(model.config.hidden_size)),\
                            'bias': torch.nn.Parameter(torch.zeros(model.config.hidden_size))})
        lens.append(nl)

    lens_model = Lens_model(lens, layers, model_name=model_name)
    lens_model.to(device)

    if dataset_train_path is None:
        dataset_train_path = "datasets/dataset_train.pkl"
    
    if dataset_test_path is None:
        dataset_test_path = "datasets/dataset_test.pkl"

    with open(dataset_train_path, "rb") as file:
        data_train = pickle.load(file)

    with open(dataset_test_path, "rb") as file:
        data_test = pickle.load(file)

    data_train.set_device(device)
    data_test.set_device(device)

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(lens_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(len(train_loader)*0.1), gamma=0.8)

    if loss_function_name == "kl":
        loss_function = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

    elif loss_function_name == "mse":
        loss_function = torch.nn.MSELoss()
    
    elif loss_function_name == "ce":
        loss_function = torch.nn.CrossEntropyLoss()

    torch.autograd.set_detect_anomaly(True)
    epoch_losses = []
    epoch_losses_valid = []
    batch_losses = []
    batch_losses_valid = []
    for epoch in range(num_epochs):
        lens_model.train()
        epoch_loss = 0
        epoch_loss_valid = 0
        print(f"Epoch: {epoch}")
        losses = []
        losses_valid = []
        for i, batch in enumerate(train_loader):
            data, input_ids, targets, attention_mask, target_index = batch
            optimizer.zero_grad()
            if loss_function_name == "kl" or loss_function_name == "mse":
                base_model_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                base_model_logits = torch.stack([base_model_logits for layer in layers], dim=-1)
                output = lens_model.forward(input_ids, attention_mask, targets)
                log_probs_model = torch.nn.functional.log_softmax(base_model_logits, dim=-2)
                log_probs_lens = torch.nn.functional.log_softmax(output, dim=-2)
                batch_size_t = target_index.shape[0]
                loss = loss_function(log_probs_lens[torch.arange(batch_size_t), target_index], log_probs_model[torch.arange(batch_size_t), target_index])

            if loss_function_name == "ce":
                output = lens_model.get_probs(input_ids, attention_mask, targets, target_index)
                batch_size_t = target_index.shape[0]
                targets_ = targets[torch.arange(batch_size_t), target_index]
                targets_ = torch.stack([targets_ for layer in layers], dim=1)
                loss = loss_function(output, targets_)

            #breakpoint()
            epoch_loss += loss.item()
            #print(loss)
            loss.backward()
            optimizer.step()
            scheduler.step()

            del data
            del input_ids
            del targets
            del attention_mask
            del target_index
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if i % int(len(train_loader)*0.01) == 0:
                print(f"    Epoch: {epoch}/{num_epochs}, Batch: {i}/{len(train_loader)}, Loss: {loss.item()}, Total Loss: {epoch_loss}")
                losses.append(loss.item())   

            if i % int(len(train_loader)*0.1) == 0:
                losses_valid_t, average_loss_valid = validate(lens_model, model, dataset_test_path, max_length, device, output_path, batch_size, loss_function_name)
                losses_valid.append(average_loss_valid)
                print(f"    Epoch: {epoch}/{num_epochs}, Batch: {i}/{len(train_loader)}, Validation Loss: {average_loss_valid}")
                epoch_loss_valid += average_loss_valid


        epoch_loss_valid /= len(train_loader)
        epoch_losses_valid.append(epoch_loss_valid)
        batch_losses_valid.append(losses_valid)        
        epoch_loss /= len(train_loader)
        print(f"Epoch: {epoch}, Loss: {epoch_loss}")
        epoch_losses.append(epoch_loss)
        batch_losses.append(losses)
        '''
        if epoch % save_interval == 0:
            with open(output_path, "wb") as file:
                pickle.dump(lens_model, file)
        '''
    
    with open(output_path, "wb") as file:
        pickle.dump(lens_model, file)

    return epoch_losses, batch_losses
    
if __name__ == "__main__":
    '''
    -model_name: str
        The name of the model to be used.
    -max_length: int
        The maximum length of the input sequence in tokens.
    -dataset_train_path: str
        The path to the training dataset.
    -dataset_test_path: str
        The path to the testing dataset.
    -device: string
        The name of device which is going to be used.
    -output_path: str
        The path to the output file.
    -layers: list of ints
        The layers to be used.
    -loss_function_name: str
        The name of the loss function to be used.
    -num_epochs: int
        The number of epochs.
    -batch_size: int
        The batch size.
    -lr: float
        The learning rate.
    -save_interval: int
        The interval of saving the model.
    
    Example:
        python train.py --model_name "gpt2" --max_length 64 --dataset_train_path "datasets/dataset_train.pkl" --dataset_test_path "datasets/dataset_test.pkl" --device "cpu" --output_path "models/lens_model.pkl" --layers 1 2 3 4 5 6 7 8 9 10 11 12 --loss_function_name "kl" --num_epochs 10 --batch_size 2 --lr 0.001 --save_interval 1
    '''
    #train("gpt2", 64, "datasets/dataset_train.pkl", "datasets/dataset_test.pkl", torch.device('cpu'), "models/lens_model.pkl", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "kl", 10, 2, 0.001, 1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--dataset_train_path", type=str, default=None)
    parser.add_argument("--dataset_test_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--layers", type=int, nargs='+', default=[-1])
    parser.add_argument("--loss_function_name", type=str, default="kl")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save_interval", type=int, default=1)
    args = parser.parse_args()
    epoch_losses, batch_losses = train(args.model_name, args.max_length, args.dataset_train_path, args.dataset_test_path, \
            torch.device(args.device), args.output_path, args.layers, args.loss_function_name,\
            args.num_epochs, args.batch_size, args.lr, args.save_interval)