from os import walk
import re
import sys
import os
import math
import numpy as np
import logging
import traceback
import json


class Utils:
    # classification tuple format = (positive/deceptive, truthful/deceptive):(1/0,1/0)
    POSITIVE_NEGATIVE_TUPLE_INDEX = 0
    TRUTHFUL_DECEPTIVE_TUPLE_INDEX = 1
    POSITIVE_CLASS_LABEL = 1
    NEGATIVE_CLASS_LABEL = -1
    TRUTHFUL_CLASS_LABEL = 1
    DECEPTIVE_CLASS_LABEL = -1
    CLASS_TUPLES = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    # model param keys
    WEIGHT_MATRIX = 'weight_matrix'
    BIAS = 'bias'
    POSITIVE_NEGATIVE = 'positive_negative'
    TRUTHFUL_DECEPTIVE = 'truthful_deceptive'
    SELECTED_FEATURES = 'selected_features'

    @classmethod
    def get_class_word_map_dict(cls):
        return {x: {} for x in cls.CLASS_TUPLES}

    @classmethod
    def get_class_count_dict(cls):
        return dict.fromkeys(cls.CLASS_TUPLES, 0)

    @staticmethod
    def get_key(key_1, key_2):
        return '{}_{}'.format(key_1, key_2)

    @staticmethod
    def update_dict_count(dict_to_update, key_to_update):
        if key_to_update not in dict_to_update:
            dict_to_update[key_to_update] = 0
        dict_to_update[key_to_update] += 1


class PerceptronClassifier:
    def __init__(self, train_data_path, epochs, num_dimensions):
        self.class_labels = Utils.CLASS_TUPLES
        self.all_word_freq_map = {}
        # self.train_data_classified = Utils.get_class_word_map_dict()
        self.class_frequency = Utils.get_class_count_dict()
        self.train_data_path = train_data_path
        self.train_data_tokenized = []
        self.train_data_label_positive_negative = []
        self.train_data_label_truthful_deceptive = []
        self.selected_feature_indices_map = {}
        self.selected_features = []
        self.train_data_vectors = []
        # self.flag = 1
        self.stop_words = self.get_stopwords()
        self.epochs = epochs
        self.num_dimensions = num_dimensions
        # self.use_averaged_perceptron = use_averaged_perceptron
        self.selected_features = tuple()
        self.create_train_data()

    def get_stopwords(self):
        with open('stopwords.txt', 'r') as stopwords_file:
            stop_words = set(stopwords_file.read().split('\n'))
        return stop_words

    def clean_input_case_text(self, file_name):
        text = open(file_name, 'r').read()
        # escape_chars = re.escape('!(),.:;?*')
        custom_strip_text = re.sub(r'[' + re.escape('!(),.:;?*') + ']', ' SpecialChars ', text)
        custom_strip_text = re.sub('[^a-z\d\s]+', ' ', custom_strip_text, flags=re.IGNORECASE)
        custom_strip_text = re.sub('(\s+)', ' ', custom_strip_text).lower()

        text_tokens = custom_strip_text.split(' ')
        cleaned_train_input = []
        for word in text_tokens:
            if word not in self.stop_words:
                cleaned_train_input.append(word)

        return cleaned_train_input

    def get_train_class(self, file_name):
        case_label = (-1, -1)
        if 'negative' in file_name and 'deceptive' in file_name:
            case_label = (Utils.NEGATIVE_CLASS_LABEL, Utils.DECEPTIVE_CLASS_LABEL)

        elif 'negative' in file_name and 'truthful' in file_name:
            case_label = (Utils.NEGATIVE_CLASS_LABEL, Utils.TRUTHFUL_CLASS_LABEL)

        elif 'positive' in file_name and 'deceptive' in file_name:
            case_label = (Utils.POSITIVE_CLASS_LABEL, Utils.DECEPTIVE_CLASS_LABEL)

        elif 'positive' in file_name and 'truthful' in file_name:
            case_label = (Utils.POSITIVE_CLASS_LABEL, Utils.TRUTHFUL_CLASS_LABEL)

        return case_label

    def create_train_data_vectors(self):
        for sample in self.train_data_tokenized:
            data_vector = [sample.get(word, 0) for word in self.selected_features]
            self.train_data_vectors.append(np.asarray(data_vector))

    def update_train_data_class(self, file_name):

        text_tokens = self.clean_input_case_text(file_name)
        text_token_freq_map = {}

        train_class_label = self.get_train_class(file_name)

        self.class_frequency[train_class_label] += 1

        for word in text_tokens:
            Utils.update_dict_count(self.all_word_freq_map, word)
            Utils.update_dict_count(text_token_freq_map, word)
            # Utils.update_dict_count(self.train_data_classified[train_class_label], word)

        self.train_data_tokenized.append(text_token_freq_map)
        self.train_data_label_positive_negative.append(train_class_label[Utils.POSITIVE_NEGATIVE_TUPLE_INDEX])
        self.train_data_label_truthful_deceptive.append(train_class_label[Utils.TRUTHFUL_DECEPTIVE_TUPLE_INDEX])

    def create_train_data(self):
        for (dir_path, dir_names, file_names) in walk(next(os.walk(self.train_data_path))[0]):
            for train_file in file_names:
                file_name = os.path.join(dir_path, train_file)
                if bool(re.search('.txt', file_name)):
                    try:
                        self.update_train_data_class(file_name)

                    except Exception as e:
                        # todo: remove before submission
                        logging.error(traceback.format_exc())

        selected_features = sorted(self.all_word_freq_map.items(), key=lambda x: x[1], reverse=True)[:self.num_dimensions]
        self.selected_features = tuple([x[0] for x in selected_features])
        self.selected_feature_indices_map = {i: x for i, x in enumerate(self.selected_features)}
        self.create_train_data_vectors()

    def train_perceptron_model(self, labels, use_averaging_perceptron=False):
        weight_matrix = np.random.rand(self.train_data_vectors[0].shape[0])
        bias = np.random.rand(1)[0]
        sample_indices = np.arange(len(self.train_data_vectors))

        if use_averaging_perceptron:
            averaging_params = np.zeros(self.train_data_vectors[0].shape[0])
            beta = 0
            count = 1

            for epoch in range(self.epochs):
                np.random.shuffle(sample_indices)
                for sample_index in sample_indices:
                    x_i = self.train_data_vectors[sample_index]
                    y_i_actual = labels[sample_index]
                    activation = y_i_actual * (np.sum(weight_matrix * x_i) + bias)
                    if activation < 0:
                        weight_matrix = weight_matrix + y_i_actual * x_i
                        bias = bias + y_i_actual
                        averaging_params = averaging_params + y_i_actual * count * x_i
                        beta = beta + y_i_actual * count
                    count += 1

            return weight_matrix - ((1 / count) * averaging_params), bias - ((1 / count) * beta)

        else:
            for epoch in range(self.epochs):
                np.random.shuffle(sample_indices)
                for sample_index in sample_indices:
                    x_i = self.train_data_vectors[sample_index]
                    y_i_actual = labels[sample_index]
                    calc_activation = y_i_actual*(np.sum(weight_matrix*x_i) + bias)

                    if calc_activation < 0:
                        weight_matrix += y_i_actual * x_i
                        bias += y_i_actual

            return weight_matrix, bias

    def train_model(self, use_averaging_perceptron=False):
        model_params = {
            Utils.SELECTED_FEATURES: self.selected_feature_indices_map
        }

        # training perceptron to classify sample as Positive or Negative
        weight_matrix, bias = self.train_perceptron_model(
            labels=self.train_data_label_positive_negative,
            use_averaging_perceptron=use_averaging_perceptron
        )
        model_params[Utils.get_key(Utils.WEIGHT_MATRIX, Utils.POSITIVE_NEGATIVE)] = list(weight_matrix)
        model_params[Utils.get_key(Utils.BIAS, Utils.POSITIVE_NEGATIVE)] = bias
        # training perceptron to classify sample as Truthful or Deceptive
        weight_matrix, bias = self.train_perceptron_model(
            labels=self.train_data_label_truthful_deceptive,
            use_averaging_perceptron=use_averaging_perceptron
        )
        model_params[Utils.get_key(Utils.WEIGHT_MATRIX, Utils.TRUTHFUL_DECEPTIVE)] = list(weight_matrix)
        model_params[Utils.get_key(Utils.BIAS, Utils.TRUTHFUL_DECEPTIVE)] = bias

        self.save_model(model_params, use_averaging_perceptron)

    def save_model(self, model_params, use_averaging_perceptron=False):
        model_file_name = "averagedmodel.txt" if use_averaging_perceptron else "vanillamodel.txt"
        with open(model_file_name, 'w') as model:
            json.dump(model_params, model)


if __name__ == '__main__':
    try:
        perceptron_classifier = PerceptronClassifier(sys.argv[1], 500, 1000)
        perceptron_classifier.train_model()
        perceptron_classifier.train_model(use_averaging_perceptron=True)

    except Exception as e:
        logging.error(traceback.format_exc())

