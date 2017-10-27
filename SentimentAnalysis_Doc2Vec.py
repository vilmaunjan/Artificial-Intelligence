# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import numpy

# classifier
from sklearn.linear_model import LogisticRegression

# random
import random

#sklearn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# For each line, it stores the list of wordsd and its label
class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    print(item_no)
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    # When training the model is better that in each epoch the sequence of sentences is randomized.
    def sentences_perm(self):
        shuffled = list(self.sentences)
        random.shuffle(shuffled)
        return shuffled

sources = {'IMDB_data/test-neg.txt': 'TEST_NEG', 'IMDB_data/test-pos.txt': 'TEST_POS',
               'IMDB_data/train-neg.txt': 'TRAIN_NEG', 'IMDB_data/train-pos.txt': 'TRAIN_POS',
               'IMDB_data/train-unsup.txt': 'TRAIN_UNS'}

sentences = LabeledLineSentence(sources)

# Model: Building the Vocabulary Table
model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7)

model.build_vocab(sentences.to_array())

# Training
for epoch in range(10):
    model.train(sentences.sentences_perm(), total_examples=model.corpus_count, epochs=model.iter)
    print(epoch)
# Inspect the model
model.most_similar('good')

model.most_similar('terrible')

#model['TRAIN_POS_0']

#Saving and Loading Models
model.save('./imdb2.d2v')

#And load it
model = Doc2Vec.load('./imdb2.d2v')

#Classifying Sentiments : use vectors to train a classifier

#create 2 parallel numpy arrays.
train_arrays = numpy.zeros((25000, 100)) #contains the vectors
train_labels = numpy.zeros(25000) #contains the labels

for i in range(12500):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_arrays[12500 + i] = model.docvecs[prefix_train_neg]
    train_labels[i] = 1
    train_labels[12500 + i] = 0

#Testing Vectors
test_arrays = numpy.zeros((40, 100))
test_labels = numpy.zeros(40)

for i in range(20):
    prefix_test_pos = 'TEST_POS_' + str(i)
    prefix_test_neg = 'TEST_NEG_' + str(i)
    test_arrays[i] = model.docvecs[prefix_test_pos]
    test_arrays[20 + i] = model.docvecs[prefix_test_neg]
    test_labels[i] = 1
    test_labels[20 + i] = 0

#Classification
classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)

print(classifier.score(test_arrays, test_labels))

predict_labels = classifier.predict(test_arrays)
#print(predict_labels)
#print(test_labels)

print( confusion_matrix(test_labels, predict_labels))
print()
print( classification_report(test_labels, predict_labels))
print()
print ("Accuracy: ", accuracy_score(test_labels, predict_labels))