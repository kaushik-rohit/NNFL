import os
import numpy as np
import keras.preprocessing.text as kpt
import keras.preprocessing.sequence as kps
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, Embedding
from sklearn.cross_validation import train_test_split
from keras.utils import to_categorical
from sklearn.utils import shuffle

def embedd_word_to_vec(X, max_word_len):
    for i in range(len(X)):
        word = X[i]
        a = np.zeros(max_word_len)
        
        for j in range(len(word)):
            if (word[j] == '-'):
                a[j] = 27
            else:
                a[j] = ord(word[j]) - 96
        X[i] = a
            
        
np.random.seed(420)

cur_dir = os.getcwd()
data_dir = os.path.join(cur_dir, 'Dataset')

with open(os.path.join(data_dir, 'adjectives.txt')) as f:
    adjectives = f.readlines()
    
with open(os.path.join(data_dir, 'verbs.txt')) as f:
    verbs = f.readlines()
    
for i in range(len(adjectives)):
    adjectives[i] = adjectives[i].rstrip('\n').lower()
    

for i in range(len(verbs)):
    verbs[i] = verbs[i].rstrip('\n').lower()


n_verbs = len(verbs)
n_adjectives = len(adjectives)
total_words = n_verbs + n_adjectives

print "No. of verbs is: %d" % (n_verbs)
print "No. of adjectives is: %d" % (n_adjectives)

X = []
Y = []

for i in range(len(adjectives)):
    X.append(adjectives[i])
    Y.append(1)
    
for i in range(len(verbs)):
    X.append(verbs[i])
    Y.append(0)
    
a = max(adjectives, key=len)
b = max(verbs, key=len)

max_word_length = max(len(a), len(b))

embedd_word_to_vec(X, max_word_length)

X = np.array(X)
Y = np.array(Y)

Y = to_categorical(Y)

print "Shape of X", X.shape
print X

X, Y = shuffle(X, Y, random_state=0)

embedding_vector_length = 23
model = Sequential()
model.add(Embedding(total_words, embedding_vector_length, input_length=max_word_length))
model.add(LSTM(10, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
model.fit(X, Y, epochs=10, batch_size=32)
# Final evaluation of the model
scores = model.evaluate(X, Y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
model.save(os.path.join(os.getcwd(), 'my_model.h5'))
