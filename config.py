

class Config(object):
    status = 'train'
    use_model = 'TextAttnBiLSTM'
    output_folder = 'output_data/'
    data_name = 'SST-2'
    SST_path = 'data/stanfordSentimentTreebank/'
    SST_pj_path = 'data/SST/'
    SST_2_path = 'data/SST-2/'
    emb_file = 'data/glove.840B.300d.txt'  # 'data/glove.6B.300d.txt'
    emb_format = 'glove'
    min_word_freq = 1
    max_len = 40

    train_limit_unit = 'SST'
    # 3, 4, 5
    # 'SST'
    test_file_1 = 'dev_SST'
    test_file_2 = 'test_origin'
    # 'test_origin', 'dev_SST'
    # 'test_{1,2,3,4,5}
