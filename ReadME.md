# Get the dataset.
You can unzip the data.7z. In addition, the datasets and corpus can also be found on [zenodo.org]{https://zenodo.org/record/3559480#.XeTMzdVG2Hs}.
# Training the Word2Vec model
then we can use w2c.py or fasttext.py to save the code after training a word2vec or fasttext model on it. 
The mode can be set to "withString" or "withoutString" through a parameter, and the tokenized data that corresponds to that setting is used as a basis for training. 
The file can contain hyperparameters such as vector dimensionality, number of iterations, and the minimum number of times a token must appear.
```
python w2v.py withString
```
# Training the LSTM model
Next, the data has to be split at random in three segments: training, validating and final testing. This script takes one argument, the data subset it should work on which was created in the previous step. 
The data is shuffled randomly and then split in parts of 80%, 10% and 10% (training, validation and final test set), and the tokens are encoded using the loaded word2vec model. 
Then, the LSTM model is trained and saved as model/model_w2v/LSTM_model.h5 and so forth.
```
python LSTM-Model-with-embedding.py
```
