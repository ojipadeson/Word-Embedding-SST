

class Config(object):
    status = 'train'
    data_name = 'SST-2'

    SST_path = 'data/stanfordSentimentTreebank/'
    SST_pj_path = 'data/SST/'

    output_folder = 'output_data/'
    emb_file = 'data/glove.840B.300d.txt'
    emb_format = 'glove'

    # output_folder = 'output_w2v/'
    # emb_file = 'data/GoogleNews-vectors-negative300.bin'
    # emb_format = 'word2vec'

    preprocess_ready = True

    min_word_freq = 1
    max_len = 40

    train_limit_unit = 'SST'
    test_file = 'dev_SST'
