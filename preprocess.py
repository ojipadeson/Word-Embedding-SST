import os
import json
import torch
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

    print('Preprocess origin : Word map, embedding pth file...')
    datasetSentences = pd.read_csv(SST_path + 'datasetSentences.txt', sep='\t')
    dictionary = pd.read_csv(SST_path + 'dictionary.txt', sep='|', header=None, names=['sentence', 'phrase ids'])
    datasetSplit = pd.read_csv(SST_path + 'datasetSplit.txt', sep=',')
    sentiment_labels = pd.read_csv(SST_path + 'sentiment_labels.txt', sep='|')

    dataset_no_crop = pd.merge(pd.merge(pd.merge(datasetSentences, datasetSplit), dictionary, on='sentence'),
                               sentiment_labels)
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

    print('Origin preprocess End\n')


def create_input_fromsst(data_name, SST_path, output_folder, max_len, mode):
    assert mode in {'train', 'test', 'dev'}

    print('Preprocess from SST --', mode)

    if mode is not 'test':
        # dataset = pd.read_csv(SST_path + mode + '.tsv', sep='\t', header=0, names=['sentence', 'sentiment_label'])
        dataset = pd.read_csv(SST_path + mode + '.tsv', sep='\t', header=0, names=['sentence', 'sentiment_label'])
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


def create_input_test(data_name, SST_path, output_folder, max_len, mode):
    print('Preprocess from SST-pj --', mode)

    dataset = pd.read_csv(SST_path + mode + '.tsv', sep='\t', header=None, names=['sentiment_label', 'sentence'])

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

    print('Preprocess from SST-pj end --', mode, '\n')


if __name__ == "__main__":
    opt = Config()
    if not opt.preprocess_ready:
        create_origin_files(data_name=opt.data_name,
                            SST_path=opt.SST_path,
                            emb_file=opt.emb_file,
                            emb_format=opt.emb_format,
                            output_folder=opt.output_folder,
                            min_word_freq=opt.min_word_freq,
                            max_len=opt.max_len)

        for file_mode in ['train', 'dev', 'test']:
            create_input_fromsst(data_name=opt.data_name,
                                 SST_path=opt.SST_pj_path,
                                 output_folder=opt.output_folder,
                                 max_len=opt.max_len,
                                 mode=file_mode)

    create_input_test(data_name=opt.data_name,
                      SST_path=opt.SST_pj_path,
                      output_folder=opt.output_folder,
                      max_len=opt.max_len,
                      mode='test_pj')
