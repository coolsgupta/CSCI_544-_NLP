import re
from collections import Counter

util_labels = ["0", "1", "2", "3"]
stop_words = set(open('stopwords.txt', 'r').read().split('\n'))
stem_word_list = {'s', 'ing'}


def get_custom_label(type):
    if type == "b":
        return {"0": {}, "1": {}, "2": {}, "3": {}}
    return dict.fromkeys(util_labels, type)


def custom_strip_text(text):
    cleaned_txt = re.sub('[^a-z\d\s]+', ' ', text, flags=re.IGNORECASE)
    return re.sub('(\s+)', ' ', cleaned_txt).lower()


def ommit_stop_words(word_arr):

    res = []
    for each in word_arr:
        if each not in stop_words:
            res.append(each)
            #res.append(each)
    return res
