# Get the dataset.
You can unzip the data.7z. In addition, the datasets and corpus can also be found on [zenodo.org](https://zenodo.org/record/3559480#.XeTMzdVG2Hs).
# Training the Word2Vec model
The word2vec model is trained on a large set of python code which is simply concatenated. If you just want to use the word2vec model, extract the prepared corpus ([pythontraining_edit.txt](https://zenodo.org/records/3559480#.XeTMzdVG2Hs)) and train it yourself using w2v.py. Otherwise, all steps from the beginning are outlined below.

Since there are syntax and indentation errors in original dataset, the following script fixes those. Note that with changes over time, different syntax errors might be introduced if the code is re-downloaded, which means that the script would need to be changed to fix those. The results are saved in pythontraining_edit.txt.
```
python w2v_cleancorpus.py
```
Next, the python tokenizer is applied to retrieve the python source code tokens. The tokenizer can be set to handle strings differently by giving the parameter "withString" or "without String". Without string would indicate that all strings are replaced by a generic string token, while the other option (with string) leaves them as they are.
```
python w2v_tokenize.py withString
```
The results of the previous step are saved as a bunch of files of the form 'pythontraining_withString.py' etc. This is because to save a lot of large files often is relatively slow, and handling them in batches is a significant improvement. The outputs are merged into a single file with the following script, which creates the file pythontraining_withString_X.py or pythontraining_withoutString_X.py, respectively.
```
python w2v_mergecorpus.py
```

then we can use w2v.py or fasttext.py to save the code after training a word2vec or fasttext model on it. 
The mode can be set to "withString" or "withoutString" through a parameter, and the tokenized data that corresponds to that setting is used as a basis for training. 
The file can contain hyperparameters such as vector dimensionality, number of iterations, and the minimum number of times a token must appear.
```
python w2v.py withString
```
# Training the LSTM model
Next, the data has to be split at random in three segments: training, validating and final testing. This script takes one argument, the data subset it should work on which was created in the previous step. 
The data is shuffled randomly and then split in parts of 70%, 15% and 15% (training, validation and final test set), and the tokens are encoded using the loaded word2vec model. 
Then, the LSTM model is trained and saved as model/model_w2v/LSTM_model.h5 and so forth.
```
python LSTM-Model-with-embedding.py
```
# Environment
cudatoolkit==10.0.130
cudnn==7.6.5
gensim==3.4.0
huggingface-hub==0.16.4
imbalanced-learn==0.9.0
keras==2.2.5
myutils==0.0.21
python==3.7.10
scikit-learn==1.0.2
tensorflow-gpu==1.15.0
torch==1.9.1+cu102
transformers==4.16.2       
xgboost==1.6.2
