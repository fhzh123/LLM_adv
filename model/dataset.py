from torch.nn import functional as F
from torch.utils.data.dataset import Dataset
import numpy as np

class Seq2Label_Dataset(Dataset):
    def __init__(self, tokenizer, src_list: list = list(), trg_list: list = None, 
                 min_len: int = 10, src_max_len: int = 300, trg_max_len: int = 300):

        self.src_tensor_list = list()
        self.trg_tensor_list = list()
        
        self.tokenizer = tokenizer

        self.min_len = min_len
        self.src_max_len = src_max_len
        self.trg_max_len = trg_max_len

        assert len(self.src_tensor_list) == len(self.trg_tensor_list)
        for src, trg in zip(src_list, trg_list):
            if min_len <= len(src):
                self.src_tensor_list.append(src)
                self.trg_tensor_list.append(trg)
        
        self.num_data = len(self.src_tensor_list)

    def __getitem__(self, index):
        src_encoded_dict = \
        self.tokenizer(
            self.src_tensor_list[index],
            max_length=self.src_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        src_input_ids = src_encoded_dict['input_ids'].squeeze(0)
        src_attention_mask = src_encoded_dict['attention_mask'].squeeze(0)
        src_token_type_ids = src_encoded_dict['token_type_ids'].squeeze(0)

        trg_label = self.trg_tensor_list[index]

        return (src_input_ids, src_attention_mask, src_token_type_ids), (trg_label)

    def __len__(self):
        return self.num_data