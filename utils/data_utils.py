import os
import numpy as np
from datasets import load_dataset

def data_split_index(data_len, valid_ratio: float = 0.1, test_ratio: float = 0.03):

    valid_num = int(data_len * valid_ratio)
    test_num = int(data_len * test_ratio)

    valid_index = np.random.choice(data_len, valid_num, replace=False)
    train_index = list(set(range(data_len)) - set(valid_index))
    test_index = np.random.choice(train_index, test_num, replace=False)
    train_index = list(set(train_index) - set(test_index))

    return train_index, valid_index, test_index

def data_load(data_path:str = None, data_name:str = None):

    total_src_list, total_trg_list = dict(), dict()

    if data_name == 'WMT2016_Multimodal':

        data_path = os.path.join(data_path,'WMT/2016/multi_modal')

        # 1) Train data load
        with open(os.path.join(data_path, 'train.de'), 'r') as f:
            total_src_list['train'] = [x.replace('\n', '') for x in f.readlines()]
        with open(os.path.join(data_path, 'train.en'), 'r') as f:
            total_trg_list['train'] = [x.replace('\n', '') for x in f.readlines()]

        # 2) Valid data load
        with open(os.path.join(data_path, 'val.de'), 'r') as f:
            total_src_list['valid'] = [x.replace('\n', '') for x in f.readlines()]
        with open(os.path.join(data_path, 'val.en'), 'r') as f:
            total_trg_list['valid'] = [x.replace('\n', '') for x in f.readlines()]

        # 3) Test data load
        with open(os.path.join(data_path, 'test.de'), 'r') as f:
            total_src_list['test'] = [x.replace('\n', '') for x in f.readlines()]
        with open(os.path.join(data_path, 'test.en'), 'r') as f:
            total_trg_list['test'] = [x.replace('\n', '') for x in f.readlines()]

    if data_name == 'IMDB':

        dataset = load_dataset("imdb")

        train_index, valid_index, _ = data_split_index(data_len=len(dataset['train']['text']), valid_ratio=0.1, test_ratio=0)

        total_src_list['train'] = np.array(dataset['train']['text'])[train_index]
        total_trg_list['train'] = np.array(dataset['train']['label'])[train_index]

        total_src_list['valid'] = np.array(dataset['train']['text'])[valid_index]
        total_trg_list['valid'] = np.array(dataset['train']['label'])[valid_index]

        total_src_list['test'] = np.array(dataset['test']['text'])
        total_trg_list['test'] = np.array(dataset['test']['label'])

    return total_src_list, total_trg_list