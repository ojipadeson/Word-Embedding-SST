import os
import json
import torch
import numpy as np
import pandas as pd
import warnings
from utils import load_embeddings
from collections import Counter
from config import Config

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None  # default='warn'


def labeling(sentiment_value, pivot=0.4):
    if sentiment_value < pivot:
        return 0
    elif sentiment_value >= 1 - pivot:
        return 1
    else:
        return -1


def check_not_punctuation(token):
    for ch in token:
        if ch.isalnum():
            return True
    return False


def filter_punctuation(s):
    s = s.lower().split(' ')
    return [token for token in s if check_not_punctuation(token)]


def tokens_to_idx(word_map, word_tokens, max_len):
    return [word_map.get(word, word_map['<unk>']) for word in word_tokens] + [word_map['<pad>']] * (
            max_len - len(word_tokens))


def create_origin_files(data_name, SST_path, emb_file, emb_format, output_folder,
                        min_word_freq, max_len):
    assert data_name in {'SST-2'}

    print('Preprocess origin datasets...')
    datasetSentences = pd.read_csv(SST_path + 'datasetSentences.txt', sep='\t')
    dictionary = pd.read_csv(SST_path + 'dictionary.txt', sep='|', header=None, names=['sentence', 'phrase ids'])
    datasetSplit = pd.read_csv(SST_path + 'datasetSplit.txt', sep=',')
    sentiment_labels = pd.read_csv(SST_path + 'sentiment_labels.txt', sep='|')

    dataset_no_crop = pd.merge(pd.merge(pd.merge(datasetSentences, datasetSplit), dictionary), sentiment_labels)
    dataset_no_crop['sentiment_label'] = dataset_no_crop['sentiment values'].apply(lambda x: labeling(x, 0.5))
    dataset_no_crop['sentence'] = dataset_no_crop['sentence'].apply(lambda s: filter_punctuation(s))

    word_freq = Counter()
    for i, tokens in enumerate(dataset_no_crop['sentence']):
        word_freq.update(tokens)
        if len(tokens) > max_len:
            dataset_no_crop['sentence'][i] = dataset_no_crop['sentence'][i][:max_len]

    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    dataset_no_crop['token_idx'] = dataset_no_crop['sentence'].apply(lambda x: tokens_to_idx(word_map, x, max_len))

    pretrain_embed, embed_dim = load_embeddings(emb_file, emb_format, word_map)
    embed = dict()
    embed['pretrain'] = pretrain_embed
    embed['dim'] = embed_dim
    torch.save(embed, output_folder + data_name + '_' + 'pretrain_embed.pth')

    with open(os.path.join(output_folder, data_name + '_' + 'wordmap.json'), 'w') as j:
        json.dump(word_map, j)

    # train_origin
    dataset_no_crop[dataset_no_crop['splitset_label'] == 1][['token_idx', 'sentiment_label']] \
        .to_csv(output_folder + data_name + '_' + 'train_origin.csv', index=False)
    # test origin
    dataset_no_crop[dataset_no_crop['splitset_label'] == 2][['token_idx', 'sentiment_label']] \
        .to_csv(output_folder + data_name + '_' + 'test_origin.csv', index=False)
    # dev origin
    dataset_no_crop[dataset_no_crop['splitset_label'] == 3][['token_idx', 'sentiment_label']] \
        .to_csv(output_folder + data_name + '_' + 'dev_origin.csv', index=False)

    print('Origin preprocess End\n')


def create_input_files(data_name, SST_path, output_folder,
                       max_len, split_percent):
    assert data_name in {'SST-2'}

    print('Split percent:', str(split_percent), 'Preprocess datasets...')
    datasetSentences = pd.read_csv(SST_path + 'datasetSentences.txt', sep='\t')
    dictionary = pd.read_csv(SST_path + 'dictionary.txt', sep='|', header=None, names=['sentence', 'phrase ids'])
    datasetSplit = pd.read_csv(SST_path + 'datasetSplit.txt', sep=',')
    sentiment_labels = pd.read_csv(SST_path + 'sentiment_labels.txt', sep='|')

    dataset = pd.merge(pd.merge(pd.merge(datasetSentences, datasetSplit), dictionary), sentiment_labels)

    # We can make a crop to dataset, but need further consideration
    dataset['sentiment_label'] = dataset['sentiment values'].apply(lambda x: labeling(x, split_percent))
    dataset = dataset[dataset['sentiment_label'] != -1]

    dataset['sentence'] = dataset['sentence'].apply(lambda s: filter_punctuation(s))

    # valid_idx = []
    for i, tokens in enumerate(dataset['sentence']):
        if len(tokens) > max_len:
            dataset['sentence'].iloc[i] = dataset['sentence'].iloc[i][:max_len]
            # valid_idx.append(i)
    # dataset = dataset.iloc[valid_idx, :]

    with open(os.path.join(output_folder, data_name + '_' + 'wordmap.json'), 'r') as json_file:
        word_map = json.load(json_file)

    dataset['token_idx'] = dataset['sentence'].apply(lambda x: tokens_to_idx(word_map, x, max_len))

    # save dataset to csv
    # train
    dataset[dataset['splitset_label'] == 1][['token_idx', 'sentiment_label']] \
        .to_csv(output_folder + data_name + '_' + 'train_' + str(split_percent)[-1] + '.csv', index=False)
    # test
    dataset[dataset['splitset_label'] == 2][['token_idx', 'sentiment_label']] \
        .to_csv(output_folder + data_name + '_' + 'test_' + str(split_percent)[-1] + '.csv', index=False)
    # dev
    dataset[dataset['splitset_label'] == 3][['token_idx', 'sentiment_label']] \
        .to_csv(output_folder + data_name + '_' + 'dev_' + str(split_percent)[-1] + '.csv', index=False)

    print('Split percent:', str(split_percent), 'Preprocess End\n')


def create_input_fromsst(data_name, SST_path, output_folder, max_len, mode):
    assert mode in {'train', 'test', 'dev'}

    print('Preprocess from SST --', mode)

    if mode is not 'test':
        # dataset = pd.read_csv(SST_path + mode + '.tsv', sep='\t', header=0, names=['sentence', 'sentiment_label'])
        dataset = pd.read_csv(SST_path + mode + '.tsv', sep='\t', header=0, names=['sentiment_label', 'sentence'])
    else:
        dataset = pd.read_csv(SST_path + mode + '.tsv', sep='\t', header=0, names=['index', 'sentence'])

    dataset['sentence'] = dataset['sentence'].apply(lambda s: filter_punctuation(s))
    for i, tokens in enumerate(dataset['sentence']):
        if len(tokens) > max_len:
            dataset['sentence'].iloc[i] = dataset['sentence'].iloc[i][:max_len]

    with open(os.path.join(output_folder, data_name + '_' + 'wordmap.json'), 'r') as json_file:
        word_map = json.load(json_file)

    dataset['token_idx'] = dataset['sentence'].apply(lambda x: tokens_to_idx(word_map, x, max_len))
    if mode is not 'test':
        dataset[['token_idx', 'sentiment_label']].to_csv(output_folder + data_name + '_' + mode + '_SST' + '.csv',
                                                         index=False)
    else:
        dataset[['token_idx']].to_csv(output_folder + data_name + '_' + mode + '_SST' + '.csv',
                                      index=False)

    print('Preprocess from SST end --', mode, '\n')


def create_test_label(SST_path, SST_2_path, mode):
    assert mode in {'train', 'test', 'dev'}

    print('Process from SST-2 --', mode)

    dataset_2_sst = pd.read_csv(SST_2_path + mode + '.tsv', sep='\t', header=0, names=['sentence', 'label'])
    dataset_pj_sst = pd.read_csv(SST_path + mode + '.tsv', sep='\t', header=0, names=['index', 'sentence'])

    dataset_pj_sst['label'] = -1

    i = 0
    for sent in dataset_pj_sst['sentence']:
        dataset_pj_sst['sentence'][i] = sent.lower()
        i += 1

    for sent in dataset_2_sst['sentence']:
        if sent in dataset_pj_sst['sentence']:
            dataset_pj_sst[dataset_pj_sst['sentence'] == sent]['label'] = dataset_2_sst['label']

    dataset_pj_sst.to_csv('SST_by_2.tsv', index=False, sep='\t')

    print('Process from SST-2 end --', mode, '\n')


def compare_same(output_folder, data_name, mode):
    dataset_sst = pd.read_csv(output_folder + data_name + '_' + mode + '_SST' + '.csv')
    dataset_origin = pd.read_csv(output_folder + data_name + '_' + mode + '_origin' + '.csv')

    merge_set = pd.merge(dataset_origin, dataset_sst, on='token_idx')
    if merge_set.shape[0] == dataset_origin.shape[0] == dataset_sst.shape[0]:
        print(mode, 'data all same', '\n')
    else:
        print(mode, 'data not same', merge_set.shape[0], dataset_origin.shape[0], dataset_sst.shape[0], '\n')


if __name__ == "__main__":
    opt = Config()
    # create_origin_files(data_name=opt.data_name,
    #                     SST_path=opt.SST_path,
    #                     emb_file=opt.emb_file,
    #                     emb_format=opt.emb_format,
    #                     output_folder=opt.output_folder,
    #                     min_word_freq=opt.min_word_freq,
    #                     max_len=opt.max_len)
    #
    # for percentage in np.arange(0.3, 0.6, 0.1):
    #     create_input_files(data_name=opt.data_name,
    #                        SST_path=opt.SST_path,
    #                        output_folder=opt.output_folder,
    #                        max_len=opt.max_len,
    #                        split_percent=percentage)
    #
    # for file_mode in ['train', 'dev', 'test']:
    #     create_input_fromsst(data_name=opt.data_name,
    #                          SST_path=opt.SST_pj_path,
    #                          output_folder=opt.output_folder,
    #                          max_len=opt.max_len,
    #                          mode=file_mode)
    #     compare_same(output_folder=opt.output_folder,
    #                  data_name=opt.data_name,
    #                  mode=file_mode)

    create_test_label(SST_path=opt.SST_pj_path,
                      SST_2_path=opt.SST_2_path,
                      mode='test')
