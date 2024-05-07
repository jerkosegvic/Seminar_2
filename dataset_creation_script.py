from gpt_2_dataset import GPT2Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pickle
import argparse

def main(model_name="gpt2", max_length=64, dataset_path=None, device=torch.device('cpu'), output_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if dataset_path is not None:
        with open(dataset_path, "rb") as file:
            data = pickle.load(file)

    else:
        data = ["Hello, how are you?", "I am fine, thank you!"]

    dataset = GPT2Dataset(data, tokenizer, max_length)

    print(f"Dataset length: {len(dataset)}")
    if output_path is not None:
        with open(output_path, "wb") as file:
            pickle.dump(dataset, file)

        print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    '''
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
    
    Example:
        python dataset_creation_script.py --model_name "gpt2" --max_length 64 --dataset_path "datasets/wikipedia_sentences_test.pkl" --device "cpu" --output_path "datasets/dataset_test.pkl"
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()
    main(args.model_name, args.max_length, args.dataset_path, torch.device(args.device), args.output_path)