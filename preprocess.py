import os
import json
import torch
import pandas as pd
import warnings
from utils import load_embeddings
from collections import Counter
from config import Config

warnings.filterwarnings("ignore")


def create_input_files(data_name, SST_path, emb_file, emb_format, output_folder, min_word_freq, max_len):
    assert data_name in {'SST-2'}

    split_percent = 0.3

    print('Preprocess datasets...')
    datasetSentences = pd.read_csv(SST_path + 'datasetSentences.txt', sep='\t')
    dictionary = pd.read_csv(SST_path + 'dictionary.txt', sep='|', header=None, names=['sentence', 'phrase ids'])
    datasetSplit = pd.read_csv(SST_path + 'datasetSplit.txt', sep=',')
    sentiment_labels = pd.read_csv(SST_path + 'sentiment_labels.txt', sep='|')

    dataset = pd.merge(pd.merge(pd.merge(datasetSentences, datasetSplit), dictionary), sentiment_labels)
    dataset_no_crop = dataset

    def labeling(sentiment_value, pivot=0.4):
        if pivot < sentiment_value < 0.5:
            return 0
        elif 0.5 <= sentiment_value < 1 - pivot:
            return 1
        else:
            return -1
        # if sentiment_value < pivot:
        #     return 0
        # elif sentiment_value >= 1 - pivot:
        #     return 1
        # else:
        #     return -1  # drop neutral

    # We can make a crop to dataset, but need further consideration
    dataset['sentiment_label'] = dataset['sentiment values'].apply(lambda x: labeling(x, split_percent))
    dataset = dataset[dataset['sentiment_label'] != -1]

    dataset_no_crop['sentiment_label'] = dataset_no_crop['sentiment values'].apply(lambda x: labeling(x, 0.5))

    def check_not_punctuation(token):
        for ch in token:
            if ch.isalnum():
                return True
        return False

    def filter_punctuation(s):
        s = s.lower().split(' ')
        return [token for token in s if check_not_punctuation(token)]

    dataset['sentence'] = dataset['sentence'].apply(lambda s: filter_punctuation(s))
    dataset_no_crop['sentence'] = dataset_no_crop['sentence'].apply(lambda s: filter_punctuation(s))

    word_freq = Counter()
    for i, tokens in enumerate(dataset_no_crop['sentence']):
        word_freq.update(tokens)
        if len(tokens) >= max_len:
            dataset_no_crop['sentence'][i] = dataset_no_crop['sentence'][i][:max_len]

    # for dataset
    valid_idx = []
    for i, tokens in enumerate(dataset['sentence']):
        if len(tokens) <= max_len:
            valid_idx.append(i)
    dataset = dataset.iloc[valid_idx, :]

    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    def tokens_to_idx(word_tokens):
        return [word_map.get(word, word_map['<unk>']) for word in word_tokens] + [word_map['<pad>']] * (
                    max_len - len(word_tokens))

    dataset['token_idx'] = dataset['sentence'].apply(lambda x: tokens_to_idx(x))
    dataset_no_crop['token_idx'] = dataset_no_crop['sentence'].apply(lambda x: tokens_to_idx(x))

    pretrain_embed, embed_dim = load_embeddings(emb_file, emb_format, word_map)
    embed = dict()
    embed['pretrain'] = pretrain_embed
    embed['dim'] = embed_dim
    torch.save(embed, output_folder + data_name + '_' + 'pretrain_embed.pth')

    with open(os.path.join(output_folder, data_name + '_' + 'wordmap.json'), 'w') as j:
        json.dump(word_map, j)

    # save dataset to csv
    # train
    split_percent = 8
    dataset[dataset['splitset_label'] == 1][['token_idx', 'sentiment_label']] \
        .to_csv(output_folder + data_name + '_' + 'train_' + str(split_percent)[-1] + '.csv', index=False)
    # test
    dataset[dataset['splitset_label'] == 2][['token_idx', 'sentiment_label']] \
        .to_csv(output_folder + data_name + '_' + 'test_' + str(split_percent)[-1] + '.csv', index=False)
    # dev
    dataset[dataset['splitset_label'] == 3][['token_idx', 'sentiment_label']] \
        .to_csv(output_folder + data_name + '_' + 'dev_' + str(split_percent)[-1] + '.csv', index=False)

    # train_origin
    dataset_no_crop[dataset_no_crop['splitset_label'] == 1][['token_idx', 'sentiment_label']] \
        .to_csv(output_folder + data_name + '_' + 'train_origin.csv', index=False)
    # test origin
    dataset_no_crop[dataset_no_crop['splitset_label'] == 2][['token_idx', 'sentiment_label']] \
        .to_csv(output_folder + data_name + '_' + 'test_origin.csv', index=False)
    # dev origin
    dataset_no_crop[dataset_no_crop['splitset_label'] == 3][['token_idx', 'sentiment_label']] \
        .to_csv(output_folder + data_name + '_' + 'dev_origin.csv', index=False)

    print('Preprocess End\n')


if __name__ == "__main__":
    opt = Config()
    create_input_files(data_name=opt.data_name,
                       SST_path=opt.SST_path,
                       emb_file=opt.emb_file,
                       emb_format=opt.emb_format,
                       output_folder=opt.output_folder,
                       min_word_freq=opt.min_word_freq,
                       max_len=opt.max_len)
