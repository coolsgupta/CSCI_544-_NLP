import sys
import math
import json
import operator
from os import walk
import os
import re


class PerceptronPredictor:
    def __init__(self, test_data_path):
        self.test_data_path = test_data_path
        self.model_file_name = 'nbmodel.txt'
        self.model = self.load_model()
        self.class_labels = ["0", "1", "2", "3"]
        self.stop_words = self.get_stopwords()
        # self.flag = 1

    def get_class(self, class_label):
        if class_label == "0":
            return "deceptive negative"

        elif class_label == "1":
            return "truthful negative"

        elif class_label == "2":
            return "truthful positive"

        return "deceptive positive"

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

    def load_model(self):
        with open(self.model_file_name) as f:
            model_params = json.load(f)
            return model_params

    def predict(self, file_name):
        text_tokens = self.clean_input_case_text(file_name)
        class_prob = self.model['class_probability']
        prior_prob = self.model['score']

        # calculate probability of each word
        text_segment_total_prob = dict.fromkeys(self.class_labels, 0)

        for word in text_tokens:
            for data_class in self.class_labels:
                if word in prior_prob[data_class]:
                    text_segment_total_prob[data_class] += prior_prob[data_class][word]

        # posterior Probability
        posterior_probability = {}
        for data_class in text_segment_total_prob.keys():
            posterior_probability[data_class] = text_segment_total_prob[data_class] + math.log(
                class_prob[data_class])

        return max(posterior_probability.items(), key=operator.itemgetter(1))[0]

    def classify(self):
        output = ''
        for (dir_path, dir_names, file_names) in walk(next(os.walk(self.test_data_path))[0]):
            for test_case in file_names:
                file_name = os.path.join(dir_path, test_case)
                if bool(re.search('.txt', file_name)):
                    predicted_class = self.get_class(self.predict(file_name))
                    predicted_class = predicted_class + " " + file_name + '\n'
                    output += predicted_class

        with open('nboutput.txt', 'w') as file:
            file.write(str(output))
            file.close()


if __name__ == '__main__':
    PerceptronPredictor(sys.argv[1]).classify()
