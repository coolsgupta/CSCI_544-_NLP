import sys
import json
from copy import deepcopy

class HMM:
    def __init__(self, argv):
        self.training_model_file = "hmmmodel.txt"
        self.results_file = "hmmoutput.txt"
        self.train_data_path = argv[1]
        self.train_data = self.create_train_data()
        self.tag_frequency_map = {'start': len(self.train_data), 'end': len(self.train_data)}
        self.word_tag_map = {}
        self.word_max_freq_tag_map = {}
        self.required_tags = []
        self.emission_prob = {}
        self.previous_tags = {}
        self.transition_prob = {}
        self.model = {}

    def create_train_data(self):
        train_data_file = open(self.train_data_path, encoding='UTF-8')
        return [line.rstrip('\n').split() for line in train_data_file]

    def get_most_frequent_tags(self):
        for each_line in self.train_data:
            for word_tag_pair in each_line:
                intermediate = word_tag_pair.split('/')
                tag = intermediate[-1]
                word = intermediate[:-1]

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
        self.required_tags = self.word_max_freq_tag_map[0:5]

    def get_emission_probability(self):
        for tag in self.word_tag_map:
            for word in self.word_tag_map[tag]:
                if word not in self.emission_prob:
                    self.emission_prob[word] = {}

                if tag not in self.emission_prob[word]:
                    self.emission_prob[word][tag] = self.word_tag_map[tag][word] / self.tag_frequency_map[tag]

    def get_previous_tags(self):
        for tags in self.tag_frequency_map:
            self.previous_tags[tags] = {}

        for line in self.train_data:
            for i in range(len(line) + 1):
                if i == 0:
                    if 'start' not in self.previous_tags[line[i].split('/')[-1]]:
                        self.previous_tags[line[i].split('/')[-1]]['start'] = 1

                    else:
                        self.previous_tags[line[i].split('/')[-1]]['start'] += 1

                elif i == len(line):
                    if line[i - 1].split('/')[-1] not in self.previous_tags['end']:
                        self.previous_tags['end'][line[i - 1].split('/')[-1]] = 1

                    else:
                        self.previous_tags['end'][line[i - 1].split('/')[-1]] += 1

                else:
                    if line[i - 1].split('/')[-1] not in self.previous_tags[line[i].split('/')[-1]]:
                        self.previous_tags[line[i].split('/')[-1]][line[i - 1].split('/')[-1]] = 1

                    else:
                        self.previous_tags[line[i].split('/')[-1]][line[i - 1].split('/')[-1]] += 1

    def get_transition_probabilities(self):
        self.transition_prob = deepcopy(self.previous_tags)
        for cur_tag in self.transition_prob:
            for prev_tag in self.tag_frequency_map:
                if prev_tag == 'end':
                    continue

                # adding add one smoothing
                elif prev_tag not in self.transition_prob[cur_tag]:
                    self.transition_prob[cur_tag][prev_tag] = 1 / (
                                self.tag_frequency_map[prev_tag] + (4 * len(self.tag_frequency_map)) - 1)
                else:
                    self.transition_prob[cur_tag][prev_tag] = (self.transition_prob[cur_tag][prev_tag] + 1) / (
                                self.tag_frequency_map[prev_tag] + (4 * len(self.tag_frequency_map)) - 1)

    def write_model(self):
        self.model = {
            'tags': self.tag_frequency_map,
            'transition': self.transition_prob,
            'emission': self.emission_prob,
            'most_tag_word': self.required_tags
        }

        with open(self.training_model_file, 'w') as model_file:
            model_file.write(json.dumps(self.model, indent=2))

    def create_model(self):
        self.get_most_frequent_tags()
        self.get_emission_probability()
        self.get_previous_tags()
        self.get_transition_probabilities()
        self.write_model()


if __name__ == '__main__':
    HMM(sys.argv).create_model()
