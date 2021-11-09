## Put data into here
* pretrained word vectors
* SST datasets

**For example**
```
├─data                
   │  glove.6B.300d.txt
   │  GoogleNews-vectors-negative300.bin
   │
   ├─stanfordSentimentTreebank
   │  datasetSentences.txt
   │  datasetSplit.txt
   │  dictionary.txt
   │  sentiment_labels.txt
   │
   └─SST
      dev.tsv
      test.tsv
      test_pj.tsv
      train.tsv
```

**If** you find your SST dataset is exactly the same with this dataset,
then you need to care nothing in this directory.

**If** you find your dataset is *SST* but the context is not same.
Don't worry, the dataset here is wide enough so that the model would probably perform well in any *SST* dataset.
*Only thing* you should do is to change ```test_pj.tsv```(with label) or ```test.tsv```(without label) file to your own test files

Format of ```test_pj.tsv```

```diff

1	effective but too-tepid biopic
1	if you sometimes like to go to the movies to have fun , wasabi is a good place to start .
1	emerges as something rare , an issue movie that 's so honest and keenly observed that it does n't feel like one .
1	the film provides some great insight into the neurotic mindset of all comics -- even those who have reached the absolute top of the game .
1	offers that rare combination of entertainment and education .
...

```

Format of ```test.tsv```

```fix

index	sentence
0	Effective but too-tepid biopic
1	If you sometimes like to go to the movies to have fun , Wasabi is a good place to start .
2	Emerges as something rare , an issue movie that 's so honest and keenly observed that it does n't feel like one .
3	The film provides some great insight into the neurotic mindset of all comics -- even those who have reached the absolute top of the game .
4	Offers that rare combination of entertainment and education .
...

```

Please refer to above formats to commit your changes.

**If** you have to use your own train dataset, change the parameter *mode* in 
function ```create_input_test()``` in [preprocess.py](../preprocess.py) from ```'test_pj'``` to ```train```,```dev```,
and run preprocess.py each time.

**If** you want to use embedding, change ```preprocess_ready``` to ```False``` in [config.py](../config.py).

**If** you want to use other datasets for embedding, feel free to contact me, any collaboration is welcomed.