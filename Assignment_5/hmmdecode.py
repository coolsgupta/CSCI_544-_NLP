import sys
import json

results_file= "hmmoutput.txt"
hmm_model_file= "hmmmodel.txt"

model_json = open('hmmmodel.txt', 'r', encoding='UTF-8')
hmm_model = json.loads(model_json.read())
model_json.close()

transition_probabilities = hmm_model['transition']
emission_probabilities = hmm_model['emission']
tag_frequency_map = hmm_model['tags']
most_common_tags = hmm_model['most_tag_word']

development_data_path = sys.argv[1]

development_data_file = open(development_data_path, 'r', encoding='UTF-8')
development_data = development_data_file.read().splitlines()

results = []

def SentenceTagging(Vmodel, wordList):
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

for sentence in development_data:
    wordList = sentence.split()
    firstWord = wordList[0]
    Vmodel = []
    Vmodel.append({})
    
    States = {}
    if firstWord in emission_probabilities.keys():
        States = emission_probabilities[firstWord]
    else:
        States = most_common_tags
    for tag in States:
        if tag == 'start' or tag=='end':
            continue
        elif firstWord in emission_probabilities:
            e_values = emission_probabilities[firstWord][tag]
        #elif tag in most_prob_tags:
        #    e_values = most_prob_tags[tag]/sum(most_prob_tags.values())
        else:
            e_values = 1  #tag_counts[tag]/sum(tag_counts.values())
        Vmodel[0][tag] = {}
        Vmodel[0][tag]['prob'] = e_values * transition_probabilities[tag]['start']
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
                    prevProb = Vmodel[-2][tag]['prob'] * transition_probabilities['end'][tag]

                    if (prevProb>maxProb['prob']):
                        maxProb['prob'] = prevProb
                        maxProb['bp'] = tag

            Vmodel[-1]['end'] = {}
            Vmodel[-1]['end']['prob'] = maxProb['prob']
            Vmodel[-1]['end']['bp'] = maxProb['bp']
        else:    
            currentWord = wordList[i]
            Vmodel.append({})
            if currentWord in emission_probabilities:
                States = emission_probabilities[currentWord]
            else:
                States = most_common_tags
            for tag in States:
                if tag=='start' or tag=='end':
                    continue
                elif currentWord in emission_probabilities:
                    e_values = emission_probabilities[currentWord][tag]
                #elif tag in most_prob_tags:
                #    e_values = most_prob_tags[tag]/sum(most_prob_tags.values())
                else:
                    e_values = 1  #tag_counts[tag]/sum(tag_counts.values())
                maxProb ={'prob':0,'bp':''}
                for lastTag in Vmodel[i-1]:
                    if lastTag=='start' or lastTag=='end':
                        continue
                    else:
                        prevProb = Vmodel[i-1][lastTag]['prob'] * e_values * transition_probabilities[tag][lastTag]

                        if(prevProb>maxProb['prob']):
                            maxProb['prob'] = prevProb
                            maxProb['bp'] = lastTag

                Vmodel[i][tag] = {}
                Vmodel[i][tag]['prob'] = maxProb['prob']
                Vmodel[i][tag]['bp'] = maxProb['bp']
    #this will append the tags sentence by sentence
    results.append(SentenceTagging(Vmodel, wordList))
                
fwrite = open('hmmoutput.txt', 'w', encoding = 'UTF-8')
for s in results:
    fwrite.write(s+'\n')