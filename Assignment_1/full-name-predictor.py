import csv


def get_last_name(name):
    names = name.lower().split('and')
    names_tokens = [[y.strip() for y in x.split()] for x in names]
    names_tokens[0].extend(names_tokens[1][len(names_tokens[0] ):])
    return ' '.join(names_tokens[0]).upper()


def get_last_name_v2(name):
    split_names = name.split(' AND ')
    names_tokens = [[y.strip() for y in x.split()] for x in split_names]
    # if len(names_tokens[0]) < len(names_tokens[1]) or len(names_tokens[0]) == 1:
    #     names_tokens[0].append(names_tokens[1][-1])
    if len(names_tokens[0]) <= 2:
        names_tokens[0].append(names_tokens[1][-1])

    return ' '.join(names_tokens[0])


class Utils:
    @staticmethod
    def read_name_file(name_file):
        with open(name_file, 'r') as name_file:
            first_names = [name_stat.split(' ')[0].strip() for name_stat in name_file.read().split('\n')]
        return first_names

    @classmethod
    def get_first_names(cls):
        names_list = []
        names_list.extend(cls.read_name_file('dist.female.first.txt'))
        names_list.extend(cls.read_name_file('dist.male.first.txt'))
        return set(names_list)


class Predictor:
    def __init__(self, test_file_path):
        self.first_names = Utils.get_first_names()
        self.test_file = test_file_path

    def check_last_name_presence(self, name_tokens):
        for i, name_token in enumerate(name_tokens):
            if name_token not in self.first_names:
                return True

        return False

    def get_last_name(self, name_tokens):
        for i, name_token in enumerate(name_tokens):
            if name_token not in self.first_names:
                return name_token[i:]

        return name_tokens[-1:]

    def predict_last_name(self, name_combi):
        names = name_combi.split(' AND ')
        names_tokens = [[name.split(' ')] for name in names]
        last_name_present = False if len(names_tokens[0]) == 1 else self.check_last_name_presence(names[0])
        predicted_name = names[0]

        if not last_name_present:
            predicted_last_name = ' '.join(self.get_last_name(names_tokens[1]))
            predicted_name = predicted_name +

        return ' '.join([names[0], predicted_last_name])


    def predict_last_names_for_names_file(self):
        predicted_names = []
        with open(self.test_file, 'r') as test_file:
            reader = csv.reader(test_file)

            correct = 0
            total = 0

            for row in reader:
                predictiction = self.predict_last_name(row[0].upper())
                predicted_names.append(predictiction)

                # todo: check code remove before submission
                total+=1
                if predictiction == row[1]:
                    correct += 1

        return predicted_names, correct, total, correct/total




if __name__ == '__main__':

