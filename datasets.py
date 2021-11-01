import time
import torch
import pandas as pd
from torch.utils.data import Dataset


class SSTreebankDataset(Dataset):
    def __init__(self, data_name, output_folder, split):
        self.split = split
        assert self.split in {'train', 'dev', 'test', 'train_origin', 'dev_origin', 'test_origin'}
        print('Loading DataSet:', self.split)
        time.sleep(0.2)

        self.dataset = pd.read_csv(output_folder + data_name + '_' + split + '.csv')

        self.dataset_size = len(self.dataset)

    def __getitem__(self, i):
        sentence = torch.LongTensor(eval(self.dataset.iloc[i]['token_idx']))  # sentence shape [max_len]
        sentence_label = self.dataset.iloc[i]['sentiment_label']

        return sentence, sentence_label

    def __len__(self):
        return self.dataset_size
