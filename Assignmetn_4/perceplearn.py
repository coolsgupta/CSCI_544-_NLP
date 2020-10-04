from os import walk
import re
import sys
import os
import math


class PerceptronClassifier:
    def __init__(self, train_data_path):
        self.class_labels = ["0", "1", "2", "3"]
        self.all_text_words = set()
        self.train_data_classified = {"0": {}, "1": {}, "2": {}, "3": {}}
        self.class_frequency = dict.fromkeys(self.class_labels, 0)
        self.train_data_path = train_data_path
        # self.flag = 1
        self.stop_words = self.get_stopwords()

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
        text_tokens = self.clean_input_case_text(file_name)

        train_class_label = self.get_train_class(file_name)

        self.class_frequency[train_class_label] += 1

        for data_class in text_tokens:
            self.all_text_words.add(data_class)
            if data_class in self.train_data_classified[train_class_label]:
                self.train_data_classified[train_class_label][data_class] += 1

            else:
                self.train_data_classified[train_class_label][data_class] = 1

    def create_train_data(self):
        for (dir_path, dir_names, file_names) in walk(next(os.walk(self.train_data_path))[0]):
            for train_file in file_names:
                file_name = os.path.join(dir_path, train_file)
                if bool(re.search('.txt', file_name)):
                    self.update_train_data_class(file_name)

    def train_model(self):
        self.create_train_data()
        num_train_cases = sum(self.class_frequency.values())
        class_probability_map = {}

        for data_class in self.class_frequency.keys():
            class_probability_map[data_class] = self.class_frequency[data_class] / num_train_cases

        len_train_words_set = len(self.all_text_words)
        train_data_keys_count = {}
        for data_class in self.class_labels:
            train_data_keys_count[data_class] = sum(self.train_data_classified[data_class].values())

        class_prob_score = {"0": {}, "1": {}, "2": {}, "3": {}}
        for word in self.all_text_words:
            for data_class in self.train_data_classified:
                count_of_word_class = (self.train_data_classified[data_class][word] + 1) if word in self.train_data_classified[data_class] else 1
                word_relative_freq_in_class = count_of_word_class / (train_data_keys_count[data_class] + len_train_words_set + 1)
                class_prob_score[data_class][word] = math.log(word_relative_freq_in_class)

        result = {"class_probability": class_probability_map, 'score': class_prob_score}
        with open('nbmodel.txt', 'w') as file:
            file.write(str(result).replace("'", "\""))
            file.close()


if __name__ == '__main__':
    PerceptronClassifier(sys.argv[1]).train_model()
