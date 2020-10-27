import sys
import json


class HMMDecode:
    def __init__(self, argv):
        self.results_file = "hmmoutput.txt"
        self.hmm_model_file = "hmmmodel.txt"
        self.hmm_model = self.get_model()
        self.tag_frequency_map = self.hmm_model['tag_frequency_map']
        self.transition_probabilities = self.hmm_model['transition_probabilities']
        self.emission_probabilities = self.hmm_model['emission_probabilities']
        self.most_common_tags = self.hmm_model['most_common_tags']
        self.development_data = self.get_development_data(argv[1])
        self.results = []

    def get_model(self):
        model_json = open(self.hmm_model_file, 'r', encoding='UTF-8')
        return json.loads(model_json.read())

    def get_development_data(self, development_data_path):
        development_data_file = open(development_data_path, 'r', encoding='UTF-8')
        return development_data_file.read().splitlines()

    def generate_sentence_tag(self, current_model, words):
        current_words_len = len(words)
        current_tag = 'end'
        result = ""
        for i in range(current_words_len - 1, -1, -1):
            result = words[i] + "/" + current_model[current_words_len][current_tag]['bp'] + " " + result
            current_tag = current_model[current_words_len][current_tag]['bp']
            current_words_len = current_words_len-1

        return result

    def write_results(self):
        fwrite = open(self.results_file, 'w', encoding='UTF-8')
        fwrite.write('\n'.join(self.results))

    def initialize_probability_map(self):
        return {'prob': 0, 'bp': ''}

    def get_current_model_max_probability(self, states, current_word, index, current_model):
        current_model.append({})
        for tag in states:
            if tag == 'start' or tag == 'end':
                continue

            if current_word in self.emission_probabilities:
                emission_probability = self.emission_probabilities[current_word][tag]

            else:
                emission_probability = 1

            max_probability = self.initialize_probability_map()
            for lastTag in current_model[index - 1]:
                if lastTag == 'start' or lastTag == 'end':
                    continue
                else:
                    previous_probability = current_model[index - 1][lastTag]['prob'] * emission_probability * \
                                           self.transition_probabilities[tag][lastTag]

                    if previous_probability > max_probability['prob']:
                        max_probability['prob'] = previous_probability
                        max_probability['bp'] = lastTag

            current_model[index][tag] = {}
            current_model[index][tag]['prob'] = max_probability['prob']
            current_model[index][tag]['bp'] = max_probability['bp']

        return current_model

    def get_current_model_end_probability(self, states, max_probability, current_model):
        current_model.append({})
        for tag in states:
            if tag == 'end':
                continue

            else:
                previous_probability = current_model[-2][tag]['prob'] * self.transition_probabilities['end'][tag]

                if previous_probability > max_probability['prob']:
                    max_probability['prob'] = previous_probability
                    max_probability['bp'] = tag

        current_model[-1]['end'] = {}
        current_model[-1]['end']['prob'] = max_probability['prob']
        current_model[-1]['end']['bp'] = max_probability['bp']
        return current_model

    def get_current_model_tag_probability(self, states, words):
        current_model = [{}]
        for tag in states:
            if tag == 'start' or tag == 'end':
                continue

            elif words[0] in self.emission_probabilities:
                emission_probability = self.emission_probabilities[words[0]][tag]

            else:
                emission_probability = 1

            current_model[0][tag] = {}
            current_model[0][tag]['prob'] = emission_probability * self.transition_probabilities[tag]['start']
            current_model[0][tag]['bp'] = 'start'

        return current_model

    def get_results(self):
        for sentence in self.development_data:
            words = sentence.split()

            if words[0] in self.emission_probabilities.keys():
                states = self.emission_probabilities[words[0]]

            else:
                states = self.most_common_tags

            current_model = self.get_current_model_tag_probability(states, words)

            for i in range(1, len(words) + 1):
                if i == len(words):
                    states = current_model[-1].keys()
                    max_probability = self.initialize_probability_map()
                    current_model = self.get_current_model_end_probability(states, max_probability, current_model)

                else:
                    current_word = words[i]
                    if current_word in self.emission_probabilities:
                        states = self.emission_probabilities[current_word]

                    else:
                        states = self.most_common_tags

                    current_model = self.get_current_model_max_probability(states, current_word, i, current_model)

            self.results.append(self.generate_sentence_tag(current_model, words))
            self.write_results()


if __name__ == '__main__':
    HMMDecode(sys.argv).get_results()
