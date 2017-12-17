import sys
import fasttext
from collections import Counter
from random import randint

output_file = './data/classifier.bin'

dim = 10
lr = 0.005
epoch = 1
min_count = 1
word_ngrams = 4
bucket = 10000000
thread = 8
silent = 1

label_prefix = '__label__'


def train_model():
    classifier = fasttext.supervised(train_file, output_file, lr=lr, epoch=epoch,min_count=min_count, word_ngrams=word_ngrams, bucket=bucket,thread=thread, silent=silent, label_prefix=label_prefix)

    result = classifier.test(test_file)
    print('Precision: {}'.format(result.precision))
    print('Recall : {}'.format(result.recall))
    print('Number of examples: {}'.format(result.nexamples))

def text_classifier(input_file):
    classifier = fasttext.load_model(output_file + '.bin')
    texts = []
    with open(input_file, 'r', errors='ignore') as input_f:
        for line in input_f:
            flag = randint(1, 10)
            if flag > 6:
                texts.append(line)
            elif len(texts) >= 30000:
                break
    labels = classifier.predict(texts)
    labels_list = []
    for label in labels:
        labels_list.append(label[0])
    most_label = max(set(labels_list), key=labels_list.count).split('__')[-1]
    print(most_label)
    return most_label
    
if __name__ == '__main__':
    input_file = sys.argv[1]
    text_classifier(input_file)
