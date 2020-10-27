import sys
import json


training_model_file= "hmmmodel.txt"
results_file= "hmmoutput.txt"

train_data_path = sys.argv[1]

train_data_file = open(train_data_path, encoding='UTF-8')

train_data = [line.rstrip('\n').split() for line in train_data_file]

tag_frequency_map = {}
word_tag_map = {}
word_max_freq_tag_map = {}

tag_frequency_map['start'] = len(train_data)
tag_frequency_map['end'] = len(train_data)

for line_list in train_data:
    for word_tag in line_list:
        intermediate = word_tag.split('/')
        tag = intermediate[-1]
        word = intermediate[:-1]
        if tag not in word_max_freq_tag_map:
            word_max_freq_tag_map[tag] = []
        if word[0] not in word_max_freq_tag_map[tag]:
            word_max_freq_tag_map[tag].append(word[0])
        word = '/'.join(word)
        if tag not in tag_frequency_map:
            tag_frequency_map[tag] = 1
        else:
            tag_frequency_map[tag]+=1
        if tag not in word_tag_map:
            word_tag_map[tag] = {}
        if word not in word_tag_map[tag]:
            word_tag_map[tag][word] = 1
        else:
            word_tag_map[tag][word]+=1
            
word_max_freq_tag_map = sorted(word_max_freq_tag_map, key=lambda k: len(word_max_freq_tag_map[k]), reverse=True)
send_tags = word_max_freq_tag_map[0:5]
            
#print(tag_word)
#print(tag_counts)
fhandle = open('hmmmodel.txt', 'w')

emission_prob = {}
for tag in word_tag_map:
    for word in word_tag_map[tag]:
        if word not in emission_prob:
            emission_prob[word] = {}
        if tag not in emission_prob[word]:
            emission_prob[word][tag] = word_tag_map[tag][word] / tag_frequency_map[tag]
            
previous_tags = {}
for tags in tag_frequency_map:
    previous_tags[tags] = {}
    
for line in train_data:
    for i in range(len(line)+1):
        if i==0:
            if 'start' not in previous_tags[line[i].split('/')[-1]]: 
                previous_tags[line[i].split('/')[-1]]['start'] = 1
            else:
                previous_tags[line[i].split('/')[-1]]['start'] += 1
        elif i==len(line):
            if line[i-1].split('/')[-1] not in previous_tags['end']:
                previous_tags['end'][line[i-1].split('/')[-1]] = 1
            else:
                previous_tags['end'][line[i-1].split('/')[-1]] += 1                
        else:
            if line[i-1].split('/')[-1] not in previous_tags[line[i].split('/')[-1]]:
                previous_tags[line[i].split('/')[-1]][line[i-1].split('/')[-1]] = 1
            else:
                previous_tags[line[i].split('/')[-1]][line[i-1].split('/')[-1]] += 1
        

transition_prob = previous_tags
#fhandle.write(json.dumps(transition_prob, indent=2))

for cur_tag in transition_prob:
    for prev_tag in tag_frequency_map:
        if prev_tag == 'end':
            continue
        #adding add one smoothing
        elif prev_tag not in transition_prob[cur_tag]:
            transition_prob[cur_tag][prev_tag] = 1/(tag_frequency_map[prev_tag] + (4 * len(tag_frequency_map)) - 1)
        else:
            transition_prob[cur_tag][prev_tag] = (transition_prob[cur_tag][prev_tag]+1)/(tag_frequency_map[prev_tag] + (4 * len(tag_frequency_map)) - 1)

model = {'tags': tag_frequency_map, 'transition': transition_prob, 'emission': emission_prob, 'most_tag_word': send_tags}

fhandle.write(json.dumps(model, indent=2))
#fhandle.write(json.dumps(transition_prob, indent=2))

fhandle.close


