import sys
import json


class HMMDecode:
    def __init__(self, argv):
        self.results_file = "hmmoutput.txt"
        self.hmm_model_file = "hmmmodel.txt"
        self.hmm_model = self.get_model()
        self.transition_probabilities = self.hmm_model['transition']
        self.emission_probabilities = self.hmm_model['emission']
        self.tag_frequency_map = self.hmm_model['tags']
        self.most_common_tags = self.hmm_model['most_tag_word']
        self.development_data = self.get_development_data(argv[1])
        self.results = []

    def get_model(self):
        model_json = open(self.hmm_model_file, 'r', encoding='UTF-8')
        return json.loads(model_json.read())

    def get_development_data(self, development_data_path):
        development_data_file = open(development_data_path, 'r', encoding='UTF-8')
        return development_data_file.read().splitlines()

    def SentenceTagging(self, Vmodel, wordList):
        curState = len(wordList)
        curTag = 'end'
        res = ""
        i = len(wordList)-1
        while i>=0:
            res = wordList[i]+"/"+Vmodel[curState][curTag]['bp']+" " + res
            curTag = Vmodel[curState][curTag]['bp']
            curState = curState-1
            i-=1

        return res

    def write_results(self):
        fwrite = open(self.results_file, 'w', encoding='UTF-8')
        fwrite.write('\n'.join(self.results))
        # for s in results:
        #     fwrite.write(s + '\n')

    def get_results(self):
        for sentence in self.development_data:
            wordList = sentence.split()
            firstWord = wordList[0]
            Vmodel = []
            Vmodel.append({})

            States = {}
            if firstWord in self.emission_probabilities.keys():
                States = self.emission_probabilities[firstWord]

            else:
                States = self.most_common_tags

            for tag in States:
                if tag == 'start' or tag=='end':
                    continue

                elif firstWord in self.emission_probabilities:
                    e_values = self.emission_probabilities[firstWord][tag]

                else:
                    e_values = 1  #tag_counts[tag]/sum(tag_counts.values())

                Vmodel[0][tag] = {}
                Vmodel[0][tag]['prob'] = e_values * self.transition_probabilities[tag]['start']
                Vmodel[0][tag]['bp'] = 'start'

            for i in range(1,len(wordList)+1):
                #handling the last step for the end state
                if i==len(wordList):
                    lastword = Vmodel[-1]
                    States = lastword.keys()
                    maxProb ={'prob':0,'bp':''}
                    Vmodel.append({})

                    for tag in States:
                        if tag=='end':
                            continue

                        else:
                            prevProb = Vmodel[-2][tag]['prob'] * self.transition_probabilities['end'][tag]

                            if (prevProb>maxProb['prob']):
                                maxProb['prob'] = prevProb
                                maxProb['bp'] = tag

                    Vmodel[-1]['end'] = {}
                    Vmodel[-1]['end']['prob'] = maxProb['prob']
                    Vmodel[-1]['end']['bp'] = maxProb['bp']

                else:
                    currentWord = wordList[i]
                    Vmodel.append({})
                    if currentWord in self.emission_probabilities:
                        States = self.emission_probabilities[currentWord]

                    else:
                        States = self.most_common_tags

                    for tag in States:
                        if tag=='start' or tag=='end':
                            continue

                        elif currentWord in self.emission_probabilities:
                            e_values = self.emission_probabilities[currentWord][tag]

                        else:
                            e_values = 1

                        maxProb ={'prob':0,'bp':''}
                        for lastTag in Vmodel[i-1]:
                            if lastTag=='start' or lastTag=='end':
                                continue
                            else:
                                prevProb = Vmodel[i-1][lastTag]['prob'] * e_values * self.transition_probabilities[tag][lastTag]

                                if(prevProb>maxProb['prob']):
                                    maxProb['prob'] = prevProb
                                    maxProb['bp'] = lastTag

                        Vmodel[i][tag] = {}
                        Vmodel[i][tag]['prob'] = maxProb['prob']
                        Vmodel[i][tag]['bp'] = maxProb['bp']

            # this will append the tags sentence by sentence
            self.results.append(self.SentenceTagging(Vmodel, wordList))
            self.write_results()


if __name__ == '__main__':
    HMMDecode(sys.argv).get_results()
