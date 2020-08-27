import csv


def get_last_name(name):
    names = name.lower().split('and')
    names_tokens = [[y.strip() for y in x.split()] for x in names]
    names_tokens[0].extend(names_tokens[1][len(names_tokens[0] ):])
    return ' '.join(names_tokens[0]).upper()


def get_last_name_v2(name):
    split_names = name.split(' AND ')
    names_tokens = [[y.strip() for y in x.split()] for x in split_names]
    if len(names_tokens[0]) < len(names_tokens[1]) or len(names_tokens[0]) == 1:
        names_tokens[0].append(names_tokens[1][-1])

    return ' '.join(names_tokens[0])


if __name__ == '__main__':
    correct = 0
    total = 0

    with open('Assignment 1//dev-key.csv') as data_file:
        reader = csv.reader(data_file)

        for row in reader:
            total += 1
            name = get_last_name(row[0])

            if name == row[1]:
                correct += 1

    print(correct*100/total)
