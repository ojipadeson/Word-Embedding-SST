# Sentiment-Analysis Using Word Embedding & LSTM & Attention

[[简体中文](README_zh.md) | [English](README.md)]

---

Use GloVe/Word2Vec embedding, LSTM and Attention to do SST sentiment classification.

---

*Word to idx is* ready in ```/output_data``` directory

---

We use 2 word embedding technique: GloVe & Word2Vec

GLoVe: We use ```glove.840B.300d.txt```. You can download it from [stanfordnlp](https://github.com/stanfordnlp/GloVe)

Word2Vec: We use ```GoogleNews-vectors-negative300.bin```. You can download .gz file from [google drive](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)

---

## Results on DEV

 Model              | Accuracy | Precision	| Recall | F1 |
 -----------------  | -----  |----- |----- |----- |
Word2Vec+LSTM+Attn  | 78.4 | 78.4 | 78.4 | 78.4 |
Word2Vec+GRU+Attn   | 79.2 | 79.2 | 79.2 | 79.2 |
GloVe+LSTM+Attn    	| 83.9 | 83.9 | 83.9 | 83.9 |
**GloVe+GRU+Attn**  | **85.1** | **85.1** | **85.0** | **85.1** |

* It's obvious that word embedding play an important roll in training
* In this project, we set large dropout for better test performance

## Run this PJ(train)

```
python main.py run
```

## Test and Get Predictions

**If** you have a ```test.tsv``` different from the one in this project's ```./data/SST``` directory, 
you should rename your ```test.tsv``` to ```test_pj.tsv```,
put it in the ```./data/SST``` directory,
and simply run:

```
python preprocess.py
```

This will use the saved word map to interpret your ```test.tsv``` to tokens.

For testing and getting predictions, simply run:
```
python main.py run --status='test' --best_model="checkpoints/BEST_checkpoint_SST-2_TextAttnBiLSTM_SST.pth"
```
The prediction for test dataset will be saved in ```./prediction.tsv```

## LICENSE
#### This is a course project
#### Reimplementation of word-embedding nlp model
#### With several more features:
* You don't need to download pre-trained word-embedding model
  if you are using SST dataset
  
* You can simply run ```main.py``` and get a model
  
* More information for model analysis metrics

## Unsatisfied with the Accuracy?

[Here](https://github.com/ojipadeson/NLP-SST-AdvanceBert) for implementation of SOTA model

---

## Acknowledgment

This project is based on [this repo@Doragd](https://github.com/Doragd/Text-Classification-PyTorch)

---