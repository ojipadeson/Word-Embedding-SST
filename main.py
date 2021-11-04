import fire
import models
from config import Config
from models.TextCNN import ModelCNN
from models.TextAttnBiLSTM import ModelAttnBiLSTM
from utils import predict_new_sample


def run(**kwargs):
    global_opt = Config()

    for k, v in kwargs.items():
        if getattr(global_opt, k, 'KeyError') != 'KeyError':
            setattr(global_opt, k, v)

    if global_opt.use_model == 'TextCNN':

        model_opt = models.TextCNN.ModelConfig()

        for k, v in kwargs.items():
            if getattr(model_opt, k, 'KeyError') != 'KeyError':
                setattr(model_opt, k, v)

        if global_opt.status == 'train':
            models.TextCNN.train_eval(model_opt)

        elif global_opt.status == 'test':
            models.TextCNN.test(model_opt)

    elif global_opt.use_model == 'TextAttnBiLSTM':

        model_opt = models.TextAttnBiLSTM.ModelConfig()

        for k, v in kwargs.items():
            if getattr(model_opt, k, 'KeyError') != 'KeyError':
                setattr(model_opt, k, v)

        if global_opt.status == 'train':
            models.TextAttnBiLSTM.train_eval(model_opt, global_opt.train_limit_unit)

        elif global_opt.status == 'test':
            models.TextAttnBiLSTM.test(model_opt, global_opt.test_file_1)
            models.TextAttnBiLSTM.test(model_opt, global_opt.test_file_2)


if __name__ == "__main__":
    fire.Fire()
