

class Config(object):
    status = 'train'
    use_model = 'TextAttnBiLSTM'
    output_folder = 'output_data/'
    data_name = 'SST-2'
    SST_path = 'data/stanfordSentimentTreebank/'
    emb_file = 'data/glove.6B.300d.txt'
    emb_format = 'glove'
    min_word_freq = 1
    max_len = 50
    train_limit_unit = 4
