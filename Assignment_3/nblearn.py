from os import walk
import re
import sys
import os
import math
from utils import *

class NaiveBayesClassifier:
    def __init__(self, train_data_path):
        self.class_labels = ["0", "1", "2", "3"]
        self.total_unique_words = set()
        self.train_data = dict.fromkeys(self.class_labels, {})
        self.class_frequency = dict.fromkeys(self.class_labels, 0)
        self.train_data_path = train_data_path
        self.flag = 1
        self.stop_words = self.get_stopwords()

    def get_stopwords(self):
        with open('stopwords.txt', 'r') as stopwords_file:
            stop_words = set(stopwords_file.read().split('\n'))
        return stop_words

    def clean_input_case_text(self, text):
        custom_strip_text = re.sub('[^a-z\d\s]+', ' ', text, flags=re.IGNORECASE)
        custom_strip_text = re.sub('(\s+)', ' ', custom_strip_text).lower()

        text_tokens = custom_strip_text.split(' ')
        cleaned_train_input = []
        for each in text_tokens:
            if each not in self.stop_words:
                cleaned_train_input.append(each)

        return cleaned_train_input

    def get_train_class(self, file_name):
        if 'negative' in file_name and 'deceptive' in file_name:
            train_class_from_directory = "0"

        elif 'negative' in file_name and 'truthful' in file_name:
            train_class_from_directory = "1"

        elif 'positive' in file_name and 'truthful' in file_name:
            train_class_from_directory = "2"

        elif 'positive' in file_name and 'deceptive' in file_name:
            train_class_from_directory = "3"

        return train_class_from_directory

    def update_train_data_class(self, file_name):
        text = open(file_name, 'r').read()
        cleaned_text = custom_strip_text(text)
        text_tokens = cleaned_text.split(' ')
        text_tokens = ommit_stop_words(text_tokens)

        if 'negative' in file_name and 'deceptive' in file_name:
            train_class_label = "0"
        elif 'negative' in file_name and 'truthful' in file_name:
            train_class_label = "1"
        elif 'positive' in file_name and 'truthful' in file_name:
            train_class_label = "2"
        elif 'positive' in file_name and 'deceptive' in file_name:
            train_class_label = "3"

        self.class_frequency[train_class_label] += 1

        for each in text_tokens:
            self.total_unique_words.add(each)
            if each in self.train_data[train_class_label]:
                self.train_data[train_class_label][each] += 1
            else:
                self.train_data[train_class_label][each] = 1


    def create_train_data(self):
        for (dir_path, dir_names, file_names) in walk(next(os.walk(self.train_data_path))[0]):
            for train_file in file_names:
                file_name = os.path.join(dir_path, train_file)
                # todo: remove below line and put re.search as re.search('.txt')
                search_str = 'fold2|fold3|fold4' if self.flag == 1 else '.txt'
                if bool(re.search(search_str, file_name)):
                    self.update_train_data_class(file_name)

    def train_model(self):
        self.create_train_data()
        total_no_of_reviews = sum(self.class_frequency.values())
        probability_of_each_class = {}
        for each in self.class_frequency.keys():
            probability_of_each_class[each] = self.class_frequency[each] / total_no_of_reviews

        mod_v = len(self.total_unique_words)
        train_data_keys = self.class_labels
        train_data_keys_count = {}
        for key in train_data_keys:
            train_data_keys_count[key] = sum(self.train_data[key].values())

        probability_score = dict.fromkeys(self.class_labels, {})
        for word in self.total_unique_words:
            for key in self.train_data:
                count_of_word_class = (self.train_data[key][word] + 1) if word in self.train_data[key] else 1
                proba_of_word_in_a_class = count_of_word_class / (train_data_keys_count[key] + mod_v + 1)
                probability_score[key][word] = math.log(proba_of_word_in_a_class)

        result = {"class_probability": probability_of_each_class, 'score': probability_score}
        with open('nbmodel.txt', 'w') as file:
            file.write(str(result).replace("'", "\""))
            file.close()


if __name__ == '__main__':
    NaiveBayesClassifier(sys.argv[1]).train_model()
