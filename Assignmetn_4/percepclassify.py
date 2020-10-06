import sys
import math
import json
import operator
from os import walk
import os
import re
import numpy as np


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

    @staticmethod
    def get_key(key_1, key_2):
        return '{}_{}'.format(key_1, key_2)

class PerceptronPredictor:
    def __init__(self, model_file_name, test_data_path):
        self.positive_negative_weight_key = Utils.get_key(Utils.WEIGHT_MATRIX, Utils.POSITIVE_NEGATIVE)
        self.truthful_deceptive_weight_key = Utils.get_key(Utils.WEIGHT_MATRIX, Utils.TRUTHFUL_DECEPTIVE)
        self.positive_negative_bias_key = Utils.get_key(Utils.BIAS, Utils.POSITIVE_NEGATIVE)
        self.truthful_deceptive_bias_key = Utils.get_key(Utils.BIAS, Utils.TRUTHFUL_DECEPTIVE)

        self.test_data_path = test_data_path
        self.stop_words = self.get_stopwords()
        self.model_params = self.load_model(model_file_name)
        self.selected_features_map = self.model_params[Utils.SELECTED_FEATURES]
        self.CLASS_LABELS = {
            (-1, -1): 'deceptive negative',
            (-1, 1): 'truthful negative',
            (1, -1): 'deceptive positive',
            (1, 1): 'truthful positive'
        }

    def get_stopwords(self):
        with open('stopwords.txt', 'r') as stopwords_file:
            stop_words = set(stopwords_file.read().split('\n'))
        return stop_words

    def get_class(self, class_label):
        return self.CLASS_LABELS.get(class_label)

    def clean_input_case_text(self, file_name):
        text = open(file_name, 'r').read()
        custom_strip_text = re.sub(r'[' + re.escape('!(),.:;?*') + ']', ' SpecialChars ', text)
        custom_strip_text = re.sub('[^a-z\d\s]+', ' ', custom_strip_text, flags=re.IGNORECASE)
        custom_strip_text = re.sub('(\s+)', ' ', custom_strip_text).lower()

        text_tokens = custom_strip_text.split(' ')
        cleaned_train_input = []
        for word in text_tokens:
            if word not in self.stop_words:
                cleaned_train_input.append(word)

        return cleaned_train_input

    def load_model(self, model_file_name):
        with open(model_file_name) as model:
            model_params = json.load(model)
            model_params[self.positive_negative_weight_key] = np.asarray(model_params[self.positive_negative_weight_key])
            model_params[self.truthful_deceptive_weight_key] = np.asarray(model_params[self.truthful_deceptive_weight_key])
            return model_params

    def create_sample_vectors(self, sample):
        data_vector = [sample.get(self.selected_features_map[str(word_index)], 0) for word_index in range(len(self.selected_features_map.keys()))]
        return np.asarray(data_vector)


    def predict(self, file_name):
        text_tokens = self.clean_input_case_text(file_name)

        text_tokens_count_map = {}
        for word in text_tokens:
            if word not in text_tokens_count_map:
                text_tokens_count_map[word] = 0
            text_tokens_count_map[word] += 1

        # vectorize sample
        text_vector = self.create_sample_vectors(text_tokens_count_map)

        prediction = [Utils.POSITIVE_CLASS_LABEL, Utils.TRUTHFUL_CLASS_LABEL]
        if np.sum(text_vector * self.model_params[self.positive_negative_weight_key]) + self.model_params[self.positive_negative_bias_key] < 0:
            prediction[Utils.POSITIVE_NEGATIVE_TUPLE_INDEX] = Utils.NEGATIVE_CLASS_LABEL

        if np.sum(text_vector * self.model_params[self.truthful_deceptive_weight_key]) + self.model_params[self.truthful_deceptive_bias_key] < 0:
            prediction[Utils.TRUTHFUL_DECEPTIVE_TUPLE_INDEX] = Utils.DECEPTIVE_CLASS_LABEL

        return self.CLASS_LABELS.get(tuple(prediction))

    def classify(self):
        output = ''
        for (dir_path, dir_names, file_names) in walk(next(os.walk(self.test_data_path))[0]):
            for test_case in file_names:
                file_name = os.path.join(dir_path, test_case)
                if bool(re.search('.txt', file_name)):
                    predicted_class = self.predict(file_name)
                    predicted_class = predicted_class + " " + file_name + '\n'
                    output += predicted_class

        with open('percepoutput.txt', 'w') as file:
            file.write(str(output))
            file.close()


if __name__ == '__main__':
    PerceptronPredictor(sys.argv[1], sys.argv[2]).classify()
