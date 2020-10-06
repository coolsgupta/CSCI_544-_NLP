from os import walk
import re
import sys
import os
import math
import numpy as np
import logging
import traceback


class Utils:
    # classification tuple format = (positive/deceptive, truthful/deceptive):(1/0,1/0)
    POSITIVE_NEGATIVE_TUPLE_INDEX = 0
    TRUTHFUL_DECEPTIVE_TUPLE_INDEX = 1
    POSITIVE_CLASS_LABEL = 1
    NEGATIVE_CLASS_LABEL = 0
    TRUTHFUL_CLASS_LABEL = 1
    DECEPTIVE_CLASS_LABEL = 0
    CLASS_TUPLES = [(0, 0), (0, 1), (1, 0), (1, 1)]

    @classmethod
    def get_class_word_map_dict(cls):
        return {x: {} for x in cls.CLASS_TUPLES}

    @classmethod
    def get_class_count_dict(cls):
        return dict.fromkeys(cls.CLASS_TUPLES, 0)

    @staticmethod
    def update_dict_count(dict_to_update, key_to_update):
        if key_to_update not in dict_to_update:
            dict_to_update[key_to_update] = 0
        dict_to_update[key_to_update] += 1


class PerceptronClassifier:
    def __init__(self, train_data_path, epochs, num_dimensions, use_averaged_perceptron=False):
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
        self.use_averaged_perceptron = use_averaged_perceptron
        self.selected_features = tuple()
        self.create_train_data()

    def get_stopwords(self):
        with open('stopwords.txt', 'r') as stopwords_file:
            stop_words = set(stopwords_file.read().split('\n'))
        return stop_words

    def clean_input_case_text(self, file_name):
        text = open(file_name, 'r').read()
        custom_strip_text = re.sub('[^a-z\d\s]+', ' ', text, flags=re.IGNORECASE)
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
        self.train_data_label_truthful_deceptive.append(train_class_label[Utils.POSITIVE_NEGATIVE_TUPLE_INDEX])

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

    def train_model(self):
        # training perceptron to classify sample as Positive or Negative

        # training perceptron to classify sample as Truthful or Deceptive
        return

if __name__ == '__main__':
    PerceptronClassifier(sys.argv[1], 500, 1000).train_model()
