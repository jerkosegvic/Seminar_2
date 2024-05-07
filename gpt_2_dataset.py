import torch
from transformers import AutoTokenizer
import copy

class GPT2Dataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length=1024, device=torch.device('cpu')):
        '''
        Initializes the GPT2Dataset class.
        data: list of strings
            The data to be tokenized.
        tokenizer: transformers.tokenizer
            The tokenizer to be used.
        max_length: int
            The maximum length of the input sequence in tokens.
        device: torch.device
            The device to be used. It can be changed later.
        '''
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.data = []
        self.input_ids = []
        self.targets = []
        self.attention_masks = []
        self.target_index = []

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        id = 0
        for entry in data:
            input_ids = self.tokenizer(entry, add_special_tokens=True, return_tensors="pt")['input_ids'][0]
            len_of_seq = len(input_ids)
            if len_of_seq > self.max_length:
                print("Too long, skipping...")
                continue
            encoding_dict = self.tokenizer(entry, max_length=self.max_length, add_special_tokens=True, padding='max_length', return_tensors="pt")
            
            input_ids = encoding_dict['input_ids']
            attention_mask = torch.full(input_ids[0].shape, 0)
            targets = torch.full(input_ids[0].shape, -100)
            
            for i in range(len_of_seq - 1):
                if (len_of_seq - i) < self.max_length:
                    attention_mask[len_of_seq - i] = 0
                    targets[len_of_seq - i] = -100
                    input_ids[0][len_of_seq - i] = tokenizer.pad_token_id

                attention_mask[len_of_seq - i - 1] = 1
                targets[len_of_seq - i - 1] = input_ids[0][len_of_seq - i - 1]
                
                input_ids_ = copy.deepcopy(input_ids)
                attention_mask_ = copy.deepcopy(attention_mask)
                targets_ = copy.deepcopy(targets)
                
                self.data.append(entry)
                self.input_ids.append(input_ids_[0])
                self.targets.append(targets_)
                self.attention_masks.append(attention_mask_)
                self.target_index.append(len_of_seq - i - 1)
                del input_ids_
                del attention_mask_
                del targets_

            del attention_mask
            del targets
            del input_ids
            del encoding_dict
            id += 1
            if id % 100 == 0 or id == 1:
                print(f"Processed {id} out of {len(data)} entries and currently there are {len(self.input_ids)} examples.")
                
    def __len__(self):
        '''
        Returns the length of the dataset.
        '''
        return len(self.data)

    def __getitem__(self, idx):
        '''
        Returns the data, input_ids, targets, attention_mask and target_index of the given index.
        '''
        if self.input_ids[idx] is not None and self.targets[idx] is not None and self.data[idx] is not None \
            and self.attention_masks[idx] is not None and self.target_index[idx] is not None:
            return self.data[idx], self.input_ids[idx].to(self.device), self.targets[idx].to(self.device), \
                self.attention_masks[idx].to(self.device), self.target_index[idx]
        
        else:
            print("None values found in the dataset")
            return [], [], [], [], -1
        
    def set_device(self, device):
        '''
        Sets the device to the given device.
        '''
        self.device = device