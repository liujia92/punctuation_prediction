import re
import sys
import os
import logging
import datetime, time
import numpy as np
import tensorflow as tf
from collections import Counter

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv1D, Embedding
from keras.models import Sequential
from keras import backend as K

import textutils

punctuation_dict = {' ': 0, ',': 1, '.': 2, '!': 3, '?': 4, ':': 5, "'": 6}
#class_weights = {0: 1, 1: 5, 2: 5, 3: 5, 4: 5, 5: 5, 6: 5}
punctuation_count = len(punctuation_dict.keys())
punctuation_str = ''.join(punctuation_dict.keys()).strip()
punctuation_space_str = ''.join(punctuation_dict.keys())

#word_embedding_file = 'glove.twitter.27B.100d.txt'
#word_embedding_file = 'glove.6B.100d.txt'
text_bin = 'classifier.bin.bin'
#words_pre_size = 10

train_size = 1000000
test_percentage = 0.8
word_index_size = 40000
validation_percentage = 0.2
embedding_dim = 100
epochs = 1

data_dir = './data'


# 获取当前时间
def current_time(time_format='%Y-%m-%d'):
    return datetime.datetime.now().strftime(time_format)

def getlog():
    logging.basicConfig(
        filename=current_time() + '.log',
        level=logging.INFO,
        format='%(asctime)s   %(levelname)s   %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    return logging

# 定义数据清理的规则
def clear_rules():
    filter_rules = [
        #(re.compile("['’]"), ","),
        #(re.compile("n't"), " n't"),
        #(re.compile("'([^t])"), " ' \g<1>"),
        #(re.compile('[-—]'), ' '),
        (re.compile("[^a-z0-9A-Z',.?!: ]"), ' '),
        (re.compile("(\\w+)([',.?!:])(\\w+)"), "\g<1>\g<2> \g<3>"),
        (re.compile("\\s+([',.?!:])\\s+"), "\g<1> "),
        (re.compile("\\s+"), " "),
        (re.compile("\\w+(\.\.\.)\\w+"), "\g<1> "),
        (re.compile("\\s(\.\.\.)\\s"), "\g<1> ")
    ]
    return filter_rules

# 清理原始数据，输出到新的文件中
def clear_data(input_file):
    # 写日志
    getlog().info('clear data {}'.format(input_file))
    # 定义输出文件名
    output_file = input_file + '.clean'
    filter_rules = clear_rules()
    with open(output_file, 'w') as output:
        with open(input_file, 'r', errors='ignore') as input:
            for line in input:
                line = line.strip()
                if line is None:
                    continue
                # 对每一行按照定义的清理规则清理
                for rules in filter_rules:
                    rule, target = rules
                    line = rule.sub(target, line)
                if len(line) == 0:
                    continue
                output.write(line + " ")
    return output_file

def write_file(output, windows, label):
    output.write(' '.join(windows))
    output.write(' ' + str(label))
    output.write('\n')

# 对数据进行格式化
# 一行中最后一位是标点符号
# 输出格式： aa bb cc dd ,
def format_data(input_file, words_pre_size):
    train_file = input_file + '.train'
    test_file = input_file + '.test'
    windows = []

    datas = []
    labels = []
    is_test_value = False

    with open(train_file, 'w') as train_f:
        with open(test_file, 'w') as test_f:
            with open(input_file, 'r') as input_f:
                count = 0
                for word in input_f.readline().split(' '):

                    if len(windows) < words_pre_size:
                        windows.append(word)
                        continue
                    if count != 0:
                        windows.append(word)
                        windows.pop(0)
                    middle = windows[int(words_pre_size/2)]
                    if re.compile('.*[{}]'.format(punctuation_str)).match(middle):
                        punctuation = middle[-1]
                        if punctuation in punctuation_str:
                            label = punctuation_dict[punctuation]
                        else:
                            label = punctuation_dict[' ']

                    else:
                        label = punctuation_dict[' ']

                    if is_test_value:
                        write_file(test_f, windows, label)
                    else:
                        datas.append(' '.join(windows))
                        labels.append(label)
                        write_file(train_f, windows, label)
                    count += 1
                    if train_size <= count:
                        break
                    elif int(train_size * test_percentage) <= count:
                        is_test_value = True
    # return datas, labels

def load_data(input_file):
    getlog().info('load data ...')
    datas = []
    labels = []
    with open(input_file, 'r') as input_f:
        count = 0
        for line in input_f:
            if line:
                line = line.rstrip()
                datas.append(line[:-1])
                labels.append(line[-1])
            if count >= train_size:
                getlog().log('load_data: {}'.format(count))
                break
        return datas, labels

# 生成一个词索引，把每个单词转换成数字，方便后续运算
def generate_word_index(datas):
    getlog().info('generate word index ...')
    tokenizer = Tokenizer(num_words=word_index_size)
    tokenizer.fit_on_texts(datas)
    word_indexs = {}
    for number, word in enumerate(tokenizer.word_index.items()):
        if number >= word_index_size - 1:
            break
        word_indexs[word[0]] = word[1]
    getlog().info('found {} unique tokens'.format(len(word_indexs)))
    np.save(os.path.join(data_dir, 'word_index.npy'), word_indexs)
    return word_indexs

def load_word_index():
    return np.load(os.path.join(data_dir, 'word_index.npy')).item()

# 把文本数据转换成对应的索引数字表示
# 如：how are you -> 23 54 23
def texts_to_sequences(word_indexs, datas, word_index_size):
    getlog().info('texts_to_sequences ... ')
    sequences = []
    for data in datas:
        word_list = text_to_word_sequence(data)
        vector = []
        for word in word_list:
            number = word_indexs.get(word)
            if number and number < word_index_size:
                vector.append(number)
            else:
                vector.append(word_index_size - 1)
        sequences.append(vector)
    return sequences


def tokenize(datas, labels, words_pre_size, word_indexs):
    getlog().info('tokenize ... ')
    tokenized_datas = texts_to_sequences(word_indexs, datas, word_index_size)
    padded_datas = pad_sequences(tokenized_datas, maxlen=words_pre_size, padding='post', truncating='post')
    tokenized_labels = to_categorical(np.asarray(labels), num_classes=punctuation_count)
    getlog().info('shape of padded_datas tensor: {}'.format(padded_datas.shape))
    getlog().info('shape of tokenized_labels tensor: {}'.format(tokenized_labels.shape))

    return padded_datas, tokenized_labels

# 把数据分成训练集和验证集
def split_train_and_validation(padded_datas, tokenized_labels):
    getlog().info('split train and validation ... ')
    indices = np.arange(padded_datas.shape[0])
    np.random.shuffle(indices)
    datas = padded_datas[indices]
    labels = tokenized_labels[indices]
    nb_validation_datas = int(validation_percentage * datas.shape[0])

    x_train = datas[:-nb_validation_datas]
    y_train = labels[:-nb_validation_datas]
    x_val = datas[-nb_validation_datas:]
    y_val = labels[-nb_validation_datas:]
    return x_train, y_train, x_val, y_val

# 使用glove数据，生成单词向量
def index_embedding_word_matrix(word_indexs):
    getlog().info('index embedding word matrix ... ')
    embedding_indexs = {}
    with open(os.path.join(data_dir, word_embedding_file), 'r') as input_f:
    #with open(os.path.join(data_dir, 'glove.twitter.27B.100d.txt'), 'r') as input_f:
        for line in input_f:
            split_data = line.split(' ')
            word = split_data[0]
            word_vector = np.asarray(split_data[1:], dtype='float32')
            embedding_indexs[word] = word_vector
    getlog().info('found {} word vectors'.format(len(embedding_indexs)))

    embedding_matrix = np.zeros((word_index_size, embedding_dim))
    count = 0
    for word, number in word_indexs.items():
        if number > word_index_size:
            continue
        embedding_vector = embedding_indexs.get(word)
        if embedding_vector is not None:
            count += 1
            embedding_matrix[number] = embedding_vector
    getlog().info('found {} word matrix'.format(count))
    return embedding_matrix


# 创建第一层 嵌入层
# 输出：word_index_size * embedding_dim
def create_embedding_layer(words_pre_size, word_indexs=None):
    getlog().info('create embedding layer ...')
    if word_indexs:
        embedding_matrix = index_embedding_word_matrix(word_indexs)
        return Embedding(input_dim=word_index_size,
                         output_dim=embedding_dim,
                         input_length=words_pre_size,
                         weights=[embedding_matrix],
                         trainable=False,
                         input_shape=(words_pre_size,))
    else:
        return Embedding(input_dim=word_index_size,
                         output_dim=embedding_dim,
                         input_length=words_pre_size,
                         trainable=False,
                         input_shape=(words_pre_size,))

# 精确率: 所有预测的结果中，正确的结果所占的比例
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, punctuation_count-1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, punctuation_count-1)))
    #getlog().info('true_positives: {}, predicted_positives: {}'.format(true_positives, predicted_positives))
    precision = true_positives / (predicted_positives + K.epsilon())
    #getlog().info('true_positives: {}, predicted_positives: {}, precision: {}'
    #      .format(true_positives, predicted_positives, precision))

    return precision

# 召回率: 预测正确的结果，占所有应该被预测正确结果的比例
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    #getlog().info('true_positives: {}, possible_positives: {}, recall: {}'
    #      .format(true_positives, possible_positives, recall))
    return recall

def fbeta_score(y_true, y_pred, beta=1):

    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 1, 4))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    getlog().info('fbeta_score: {}'.format(fbeta_score))
    return fbeta_score

def init_model_variable(types):
    word_pre_size = 10
    kernel_size = 3
    activation = 'softmax'
    dropout = 0.2
    optimizer = 'adam'
    word_embedding_file = 'glove.6B.100d.txt'
    if 'kika' == types:
        kernel_size = 10
        dropout = 0.1
        activation = 'softmax'
        word_embedding_file = 'glove.twitter.27B.100d.txt'
    elif 'wiki' == types:
        word_pre_size = 20
        kernel_size = 10
        dropout = 0.3
    elif 'twitter' == types:
        word_pre_size = 15
        kernel_size = 5
        dropout = 0.4
        word_embedding_file = 'glove.twitter.27B.100d.txt'
    elif 'common' == types:
        word_pre_size = 15
        kernel_size = 5
        dropout = 0.3
    return word_pre_size, kernel_size, activation, dropout, optimizer, word_embedding_file

# 创建模型
def create_model(inits, word_indexs=None):

    getlog().info('text types : {}'.format(types))
    getlog().info('create model ... ')

    words_pre_size, kernel_size, activation, dropout, optimizer, word_embedding_file = inits

    model = Sequential()

    # 输出 word_index_size * embedding_dim
    model.add(create_embedding_layer(words_pre_size, word_indexs))
    # 输出 word_index_size * 512
    model.add(Conv1D(filters=512, kernel_size=kernel_size, activation='relu', use_bias=True))
    if word_indexs:
        model.add(Dropout(dropout))
    #model.add(Dense(units=128, activation='relu', use_bias=True))
    # 如果word_indexs＝None说明是测试
    #if word_indexs:
    #    model.add(Dropout(0.4))
    model.add(Flatten())
    # 输出 word_index_size * punctuation_count
    #model.add(Dense(units=punctuation_count, activation='sigmoid', use_bias=True))
    model.add(Dense(units=punctuation_count, activation=activation, use_bias=True))
    # alternative optimizer: rmsprop, adam
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy', precision, recall, fbeta_score])

    return model

def train_model(model, x_train, y_train, x_val, y_val):
    getlog().info('train model...')
    #hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=128, class_weight=class_weights)
    hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=128)
    pred = model.predict_classes(x_train, batch_size=128)

    model.save_weights(os.path.join(data_dir, 'model_weights.txt'))
    history = hist.history.keys()
    with open(os.path.join(data_dir, 'model_processing.log'), 'w') as output_f:
        output_f.write('history : {}'.format(history))
    return pred.tolist()

def test_model(inits, input_file):
    words_pre_size, kernel_size, activation, dropout, optimizer, word_embedding_file = inits
    getlog().info('test model...')
    test_datas, test_labels = load_data(input_file)
    word_indexs = load_word_index()
    model = create_model((words_pre_size, kernel_size, activation, dropout, optimizer, word_embedding_file))
    model.load_weights(os.path.join(data_dir, 'model_weights.txt'))
    test_padded_datas, test_tokenized_labels = tokenize(test_datas, test_labels, words_pre_size, word_indexs)
    metrics_values = model.evaluate(test_padded_datas, test_tokenized_labels, 128)
    pred = model.predict_classes(test_padded_datas, batch_size=128)
    getlog().info('metrics_name: {}, metrics_values: {}'.format(model.metrics_names, metrics_values))
    # predict = lambda tokenized: model.predict(tokenized)[0]
    return test_tokenized_labels, pred

def array_to_list(labels):
    """ labels: [[ 1.  0.  0.  0.  0.  0.]
                [ 1.  0.  0.  0.  0.  0.]
                [ 0.  0.  0.  0.  1.  0.]
                [ 1.  0.  0.  0.  0.  0.]]
    转换成: [0, 0, 4, 0]
    """
    labels_list = labels.tolist()
    results = []
    for labels in labels_list:
        results.append(labels.index(1))
    getlog().info(results[:10])
    return results

def compute_acc(y_true, y_pred):
    results = {}
    fail_results = {}
    y_len = len(y_true)

    for i in range(0, y_len):
        if y_true[i] == y_pred[i]:
            if y_true[i] in results:
                true_count = results[y_true[i]]
                results[y_true[i]] = true_count + 1
            else:
                results[y_true[i]] = 1
        else:
            if y_true[i] in fail_results:
                if y_pred[i] in fail_results[y_true[i]]:
                    fail_count = fail_results[y_true[i]][y_pred[i]]
                    fail_results[y_true[i]][y_pred[i]] = fail_count + 1
                else:
                    fail_results[y_true[i]][y_pred[i]] = 1
            else:
                temp_d = {}
                temp_d[y_pred[i]] = 1
                fail_results[y_true[i]] = temp_d
    # 真实值 和 预测值 中每个标点的个数
    y_true_dict = Counter(y_true)
    y_pred_dict = Counter(y_pred)
    # 对应的比例
    y_true_dict_rate = {}
    y_pred_dict_rate = {}
    for r in y_true_dict.keys():
        y_true_dict_rate[r] = str(round(y_true_dict[r] / y_len * 100, 2)) + '%'
        y_pred_dict_rate[r] = str(round(y_pred_dict[r] / y_len * 100, 2)) + '%'

    getlog().info('真实值的标点符号个数: {}'.format(y_true_dict))
    getlog().info('真实值的标点符号占比: {}'.format(y_true_dict_rate))
    getlog().info('预测值的标点符号个数: {}'.format(y_pred_dict))
    getlog().info('预测值的标点符号占比: {}'.format(y_pred_dict_rate))
    print('真实值的标点符号个数: {}'.format(y_true_dict))
    print('真实值的标点符号占比: {}'.format(y_true_dict_rate))
    print('预测值的标点符号个数: {}'.format(y_pred_dict))
    print('预测值的标点符号占比: {}'.format(y_pred_dict_rate))

    getlog().info('预测对的结果: {}'.format(results))
    print('预测对的结果: {}'.format(results))
    getlog().info('预测错的结果: {}'.format(fail_results))
    print('预测错的结果: {}'.format(fail_results))

    try:
        # 计算 总体标点符号的精确率, 正确的标点符号总个数 / 预测的所有标点符号的总个数
        true_total = 0
        for num in y_true_dict.keys():
            if num > 0:
                true_total = true_total + y_true_dict[num]
        pred_total = 0
        for num in y_pred_dict.keys():
            if num > 0:
                pred_total = pred_total + y_pred_dict[num]
        pred_true_total = 0
        for num in results.keys():
            if num > 0:
                pred_true_total = pred_true_total + results[num]
        precision_total = str(round(pred_true_total / pred_total * 100, 2)) + '%'
        recall_total = str(round(pred_true_total / true_total * 100, 2)) + '%'
        print('总体的 precision : {}'.format(precision_total))
        print('总体的 recall : {}'.format(recall_total))
        getlog().info('总体的 precision : {}'.format(precision_total))
        getlog().info('总体的 recall : {}'.format(recall_total))
        # 计算 每个标点的精确率, 正确的标点符号个数 / 预测的所有该标点符号的个数
        precision_dict = {}
        for label in y_pred_dict.keys():
            try:
                precision_dict[label] = str(round(results[label] / y_pred_dict[label] * 100, 2)) + '%'
            except Exception as e:
                getlog().info('precision 计算报错: {}'.format(e))
                print('precision 计算报错: {}'.format(e))
                continue
        print('标点符号的precision : {}'.format(precision_dict))
        getlog().info('标点符号的precision : {}'.format(precision_dict))

        # 计算 每个标点的recall, 预测对该标点符号的个数 ／ 该标点符号的总数
        recall_dict = {}
        for label in y_true_dict.keys():
            try:
                recall_dict[label] = str(round(results[label] / y_true_dict[label] * 100, 2)) + '%'
            except Exception as e:
                getlog().info('recall 计算报错: {}'.format(e))
                print('recall 计算报错: {}'.format(e))
                continue
        print('标点符号的recall : {}'.format(recall_dict))
        getlog().info('标点符号的recall : {}'.format(recall_dict))
    except Exception as e:
        getlog().info('计算报错: {}'.format(e))
        print('计算报错: {}'.format(e))

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('python3 model2.py twitter "update active = sigmod"')
        exit(1)
    types = sys.argv[1]
    update = sys.argv[2]
    getlog().info('----------{}----------'.format(update))
    #data_file = os.path.join(data_dir, 'mix_all.log')
    #data_file = os.path.join(data_dir, 'mix_all_2.log')
    #data_file = os.path.join(data_dir, 'en_punctuation_recommend_train_100W')
    
    data_file = os.path.join(data_dir, 'data_{}.txt'.format(types))
    # 获取文档类型
    classifier = textutils.text_classifier(data_file)
    # 通过文本类型,初始化模型参数
    words_pre_size, kernel_size, activation, dropout, optimizer, word_embedding_file = init_model_variable(classifier)
    # 清理数据
    clear_data(data_file)
    # 格式化数据
    format_data(data_file + '.clean', words_pre_size)
    datas, labels = load_data(data_file + '.clean.train')
    word_indexs = generate_word_index(datas)
    word_indexs = load_word_index()
    padded_datas, tokenized_labels = tokenize(datas, labels, words_pre_size, word_indexs)
    x_train, y_train, x_val, y_val = split_train_and_validation(padded_datas, tokenized_labels)
    model = create_model((words_pre_size, kernel_size, activation, dropout, optimizer, word_embedding_file), word_indexs)
    y_pred = train_model(model, x_train, y_train, x_val, y_val)
    compute_acc(array_to_list(y_train), y_pred)
    y_test, y_pred = test_model((words_pre_size, kernel_size, activation, dropout, optimizer, word_embedding_file), data_file + '.clean.test')
    #y_test, y_pred = test_model('./data/en_punctuation_recommend_train_100W.clean.test')
    compute_acc(array_to_list(y_test), y_pred)

