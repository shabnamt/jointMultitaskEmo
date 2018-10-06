'''
Author: Shabnam Tafreshi
Date: Sep. 7th 2018

This code is modified version of the system we proposed in the following accepted paper:
Emotion Detection and Classification in a Multigenre Corpus with Joint Multi-Task Deep Learning, COLING 2018
However, this code does not replicate the results due to one of the data set not being available publicly.
This code is available free for non-commercial use.
Please cite this paper if you use this code:

@inproceedings{tafreshi2018emotion,
  title={Emotion Detection and Classification in a Multigenre Corpus with Joint Multi-Task Deep Learning},
  author={Tafreshi, Shabnam and Diab, Mona},
  booktitle={Proceedings of the 27th International Conference on Computational Linguistics},
  pages={2905--2913},
  year={2018}
}

**** Please follow the code where it points you to "important notes" ****
'''

__author__ = 'Shabnam Tafreshi'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.utils import np_utils
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import concatenate
from keras.layers.embeddings import Embedding
from keras.layers import GRU
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import classification_report
import pickle

'''
******* IMPORTANT NOTE ********
Due to tweet data not being available to us, this code does not
replicate the results from our paper. Hence, if you wish to use this code
please change the following variables:

'''

# Variables initiation
task1denselayer = 30 #numner of unit in dense layer in task-1
task2denselayer = 20 #numner of unit in dense layer in task-2
batchsize = 5 #batchsize in traning
numofwords = 300  # maximum number of words to keep from corpus, change it according to yours
max_len = 70  # length of each sentence when indexing and zeropadding
embeddingsize = 300  # dimension of embedding layer


# ************ Reading Training Files ***************

print("Openning Training Files...")
# IMPORTANT NOTE: Change this file name to your training text and label filename
train_text = "./training_text.txt"
train_label = "./training_label.txt"

with open(train_text, encoding="utf8") as sentences:
    text_data = []
    for sentence in sentences:
        text_data.append(sentence)

with open(train_label, "r") as labels:
    labels_data = []
    for label in labels:
        labels_data.append(label)


# ************ Tokenzing and sequencing **************
print("Tokenizing... Fitting... Sequencing... \n")

tokenizer = Tokenizer(num_words=numofwords, lower=True, split=" ")
# Fit tokenizing
tokenizer.fit_on_texts(text_data)
# Text file sequencing
mixgenre_sequences = tokenizer.texts_to_sequences(text_data)

# Text file indexes
word_index = tokenizer.word_index

with open('./jmte_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# ************ Shaping Training Data *****************
print("Shaping Train Data...\n")

train_labels_categorical = np_utils.to_categorical(np.asarray(labels_data))
train_mixgenre_seq_pad = pad_sequences(mixgenre_sequences, maxlen=max_len)

t1_y_train = train_labels_categorical[0:69]
t1_x_train = train_mixgenre_seq_pad[0:69]


# Task-2 training data and labels
t2_y_train = train_labels_categorical[70:139]
t2_x_train = train_mixgenre_seq_pad[70:139]


# *********** Building The Model *********************
''' 
******* IMPORTANT NOTE ********
1. The following indexes reflect our sample set split, please change them according to your split
2. Both train and test sets should have the same matrix size otherwise the code will not run
'''

print("Building model...\n")

print("Building Embedding layer....\n")

# Shared layers
optadam = Adam(lr=0.001)
t1_input_layer = Input(shape=(max_len,))
t2_input_layer = Input(shape=(max_len,))

shared_emb_layer = Embedding(len(word_index) + 1, embeddingsize, input_length=max_len, trainable=True)
t1_emb_layer = shared_emb_layer(t1_input_layer)
t2_emb_layer = shared_emb_layer(t2_input_layer)

shared_grnn_layer = GRU(max_len, activation='relu')
t1_grnn_layer = shared_grnn_layer(t1_emb_layer)
t2_grnn_layer = shared_grnn_layer(t2_emb_layer)

# Merging layers
merge_layer = concatenate([t1_grnn_layer, t2_grnn_layer], axis=-1)

# Task-1 Specified Layers
t1_dense_1 = Dense(task1denselayer, activation='relu')(merge_layer)
t1_dropout_layer = Dropout(0.3)(t1_dense_1)
t1_dense_2 = Dense(task1denselayer, activation='relu')(t1_dropout_layer)
t1_dense_3 = Dense(task1denselayer, activation='relu')(t1_dense_2)
t1_prediction = Dense(8, activation='softmax')(t1_dense_3)

# Task-2 Specified Layers
t2_dense_1 = Dense(task2denselayer, activation='relu')(merge_layer)
t2_dropout_layer = Dropout(0.3)(t2_dense_1)
t2_dense_2 = Dense(task2denselayer, activation='relu')(t2_dropout_layer)
t2_dense_3 = Dense(task2denselayer, activation='relu')(t2_dense_2)
t2_prediction = Dense(8, activation='softmax')(t2_dense_3)

# Build the model
multitask_model = Model(inputs=[t1_input_layer, t2_input_layer],
                        outputs=[t1_prediction, t2_prediction])
# Compile the model
multitask_model.compile(optimizer=optadam, loss='categorical_crossentropy', metrics=['accuracy'])

# Printing summery of the layers
print(multitask_model.summary())

# Fitting the model
multitask_model.fit([t1_x_train, t2_x_train], [t1_y_train, t2_y_train], epochs=10, batch_size=batchsize, verbose=2)

# *************** Prediction & Evaluation ****************


# *************** Opening Test Files
print("Openning Test File...")
# IMPORTANT NOTE: Change this file name to your training text and label filename
test_text = "./test_text.txt"

with open(test_text, encoding="utf8") as sentences:
    test_text = []
    for sentence in sentences:
        test_text.append(sentence)

tokenizer = Tokenizer()
with open('./jmte_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
'''
******* IMPORTANT NOTE ********
1. The following indexes reflect our sample set split, please change them according to your split
2. Both train and test sets should have the same matrix size otherwise the code will not run
'''
test_mixgenre_sequences = tokenizer.texts_to_sequences(test_text)
test_mixgenre_seq_pad = pad_sequences(test_mixgenre_sequences, maxlen=max_len)
# Task-1 test data and labels
t1_x_test = test_mixgenre_seq_pad[0:29]

# Task-2 test data and labels
t2_x_test = test_mixgenre_seq_pad[30:59]

# Predicting the tag
yhat = multitask_model.predict([t1_x_test, t2_x_test], verbose=0)
t1_labels = np.argmax(yhat[0], axis=-1)
t2_labels = np.argmax(yhat[1], axis=-1)

# Writing the results on a file
t1results = open("./task_1_noaux_results.txt", "w")
t2results = open("./task_2_noaux_results.txt", "w")
for i in range(0, len(t1_x_test)):
    tag = ''
    if str(t1_labels[i]) == '0':
        tag = 'anger'
    elif str(t1_labels[i]) == '1':
        tag = 'anticipation'
    elif str(t1_labels[i]) == '2':
        tag = 'disgust'
    elif str(t1_labels[i]) == '3':
        tag = 'fear'
    elif str(t1_labels[i]) == '4':
        tag = 'joy'
    elif str(t1_labels[i]) == '5':
        tag = 'sadness'
    elif str(t1_labels[i]) == '6':
        tag = 'surprise'
    elif str(t1_labels[i]) == '7':
        tag = 'trust'
    t1results.write(tag + '\n')

for i in range(0, len(t2_x_test)):
    tag = ''
    if str(t2_labels[i]) == '0':
        tag = 'anger'
    elif str(t2_labels[i]) == '1':
        tag = 'anticipation'
    elif str(t2_labels[i]) == '2':
        tag = 'disgust'
    elif str(t2_labels[i]) == '3':
        tag = 'fear'
    elif str(t2_labels[i]) == '4':
        tag = 'joy'
    elif str(t2_labels[i]) == '5':
        tag = 'sadness'
    elif str(t2_labels[i]) == '6':
        tag = 'surprise'
    elif str(t2_labels[i]) == '7':
        tag = 'trust'
    t2results.write(tag + '\n')

t1results.close()
t2results.close()
# **************** Evaluating the model if you have gold labels ***************

# IMPORTANT NOTE: if you do not have evaluation set, you may comment from this point onward

print("Openning Test Labels...")

# IMPORTANT NOTE: Change this file name to your test filename
test_label = "./test_label.txt"

with open(test_label, "r") as labels:
    test_labels = []
    for label in labels:
        test_labels.append(label)

labels_categorical = np_utils.to_categorical(np.asarray(test_labels))


'''
******* IMPORTANT NOTE ********
1. The following indexes reflect our sample set split, please change them according to your split
2. Both train and test sets should have the same matrix size otherwise the code will not run
'''

# Task-1 test data and labels
t1_y_test = labels_categorical[0:29]

# Task-2 test data and labels
t2_y_test = labels_categorical[30:59]

multitask_scores = multitask_model.evaluate([t1_x_test, t2_x_test], [t1_y_test, t2_y_test], verbose=0)
print("Accuracy on task-1: %.2f%%" % (multitask_scores[3]*100))
print("Accuracy on task-2: %.2f%% \n" % (multitask_scores[4]*100))

t1_Y_test = np.argmax(t1_y_test, axis=-1)
t1_yhat = np.argmax(yhat[0], axis=-1)
print("\n Task-1 results:")
print(classification_report(t1_Y_test, t1_yhat))

t2_Y_test = np.argmax(t2_y_test, axis=-1)
t2_yhat = np.argmax(yhat[1], axis=-1)
print("\n Task-2 results:")
print(classification_report(t2_Y_test, t2_yhat))

print("ALL DONE!!!")
