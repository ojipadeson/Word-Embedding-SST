import fire
import models
from config import Config
from models.TextAttnBiLSTM import ModelAttnBiLSTM


def run(**kwargs):
    global_opt = Config()

    for k, v in kwargs.items():
        if getattr(global_opt, k, 'KeyError') != 'KeyError':
            setattr(global_opt, k, v)

    model_opt = models.TextAttnBiLSTM.ModelConfig()

    for k, v in kwargs.items():
        if getattr(model_opt, k, 'KeyError') != 'KeyError':
            setattr(model_opt, k, v)

    if global_opt.status == 'train':
        models.TextAttnBiLSTM.train_eval(model_opt, global_opt.train_limit_unit)

    elif global_opt.status == 'test':
        models.TextAttnBiLSTM.test(model_opt, global_opt.test_file, True)


if __name__ == "__main__":
    fire.Fire()
