import nltk
from gensim.models import Word2Vec, KeyedVectors
import os.path
import pickle
import sys

all_words = []
path = "/home/extra_space/xsl/EmbeddingVer/"

mode = "withString"  # default
if len(sys.argv) > 1:
    mode = sys.argv[1]

# Loading the training corpus
print("Loading " + mode)
with open(path + 'data/pythontraining' + '_' + mode + "_X", 'r') as file:
    pythondata = file.read().lower().replace('\n', ' ')

print("Length of the training file: " + str(len(pythondata)) + ".")
print("It contains " + str(pythondata.count(" ")) + " individual code tokens.")

# Preparing the dataset (or loading already processed dataset to not do everything again)
if os.path.isfile(path + 'data/pythontraining_processed_' + mode):
    with open(path + 'data/pythontraining_processed_' + mode, 'rb') as fp:
        all_words = pickle.load(fp)
    print("loaded processed model.")
else:
    print("now processing...")
    processed = pythondata
    all_sentences = nltk.sent_tokenize(processed)
    all_words = [nltk.word_tokenize(sent) for sent in all_sentences]
    print("saving")
    with open(path + 'data/pythontraining_processed_' + mode, 'wb') as fp:
        pickle.dump(all_words, fp)

print("processed.\n")

# trying out different parameters
# trying out different parameters
for mincount in [10]:
    for iterationen in [100]:
        for s in [200]:

            print("\n\n" + mode + " W2V model with min count " + str(mincount) + " and " + str(
                iterationen) + " Iterationen and size " + str(s))
            fname = path + "Word2v-embbeding/word2vec_" + mode + str(mincount) + "-" + str(iterationen) + "-" + str(
                s) + ".model"

            if os.path.isfile(fname):
                print("model already exists.")
                continue

            else:
                print("calculating model...")
                # training the model
                model = Word2Vec(all_words, size=s, min_count=mincount, iter=iterationen, workers=4)
                vocabulary = model.wv.vocab

                model.save(fname)
