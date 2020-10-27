import sys
import json
from copy import deepcopy


class HMM:
    def __init__(self, argv):
        self.training_model_file = "hmmmodel.txt"
        self.train_data_path = argv[1]
        self.train_data = self.create_train_data()
        self.tag_frequency_map = {'start': len(self.train_data), 'end': len(self.train_data)}
        self.word_tag_map = {}
        self.word_max_freq_tag_map = {}
        self.most_common_tags = []
        self.emission_probabilities = {}
        self.transition_probabilities = {}
        self.model = {}

    def create_train_data(self):
        train_data_file = open(self.train_data_path, encoding='UTF-8')
        return [line.rstrip('\n').split() for line in train_data_file]

    def get_most_frequent_tags(self):
        for each_line in self.train_data:
            for word_tag_pair in each_line:
                word_tag_pair_split = word_tag_pair.split('/')
                tag = word_tag_pair_split[-1]
                word = word_tag_pair_split[:-1]

                if tag not in self.word_max_freq_tag_map:
                    self.word_max_freq_tag_map[tag] = []

                if word[0] not in self.word_max_freq_tag_map[tag]:
                    self.word_max_freq_tag_map[tag].append(word[0])

                word = '/'.join(word)
                if tag not in self.tag_frequency_map:
                    self.tag_frequency_map[tag] = 1

                else:
                    self.tag_frequency_map[tag] += 1

                if tag not in self.word_tag_map:
                    self.word_tag_map[tag] = {}

                if word not in self.word_tag_map[tag]:
                    self.word_tag_map[tag][word] = 1

                else:
                    self.word_tag_map[tag][word] += 1

        self.word_max_freq_tag_map = sorted(
            self.word_max_freq_tag_map,
            key=lambda k: len(self.word_max_freq_tag_map[k]),
            reverse=True
        )
        self.most_common_tags = self.word_max_freq_tag_map[0:5]

    def get_emission_probability(self):
        for tag in self.word_tag_map:
            for word in self.word_tag_map[tag]:
                if word not in self.emission_probabilities:
                    self.emission_probabilities[word] = {}

                if tag not in self.emission_probabilities[word]:
                    self.emission_probabilities[word][tag] = self.word_tag_map[tag][word] / self.tag_frequency_map[tag]

    def calculate_transition_probabilities(self):
        for cur_tag in self.transition_probabilities:
            for prev_tag in self.tag_frequency_map:
                if prev_tag == 'end':
                    continue

                elif prev_tag not in self.transition_probabilities[cur_tag]:
                    self.transition_probabilities[cur_tag][prev_tag] = 1 / (self.tag_frequency_map[prev_tag] + (4 * len(self.tag_frequency_map)) - 1)

                else:
                    self.transition_probabilities[cur_tag][prev_tag] = (self.transition_probabilities[cur_tag][prev_tag] + 1) / (self.tag_frequency_map[prev_tag] + (4 * len(self.tag_frequency_map)) - 1)

    def get_transition_probabilities(self):
        for tags in self.tag_frequency_map:
            self.transition_probabilities[tags] = {}

        for line in self.train_data:
            for i in range(len(line) + 1):
                if i == 0:
                    if 'start' not in self.transition_probabilities[line[i].split('/')[-1]]:
                        self.transition_probabilities[line[i].split('/')[-1]]['start'] = 1

                    else:
                        self.transition_probabilities[line[i].split('/')[-1]]['start'] += 1

                elif i == len(line):
                    if line[i - 1].split('/')[-1] not in self.transition_probabilities['end']:
                        self.transition_probabilities['end'][line[i - 1].split('/')[-1]] = 1

                    else:
                        self.transition_probabilities['end'][line[i - 1].split('/')[-1]] += 1

                else:
                    if line[i - 1].split('/')[-1] not in self.transition_probabilities[line[i].split('/')[-1]]:
                        self.transition_probabilities[line[i].split('/')[-1]][line[i - 1].split('/')[-1]] = 1

                    else:
                        self.transition_probabilities[line[i].split('/')[-1]][line[i - 1].split('/')[-1]] += 1

        self.calculate_transition_probabilities()

    def write_model(self):
        self.model = {
            'tag_frequency_map': self.tag_frequency_map,
            'transition_probabilities': self.transition_probabilities,
            'emission_probabilities': self.emission_probabilities,
            'most_common_tags': self.most_common_tags
        }

        with open(self.training_model_file, 'w') as model_file:
            model_file.write(json.dumps(self.model, indent=2))

    def create_model(self):
        self.get_most_frequent_tags()
        self.get_emission_probability()
        self.get_transition_probabilities()
        self.write_model()


if __name__ == '__main__':
    HMM(sys.argv).create_model()
