import csv
import numpy as np

def load_train_data():
    train_file_path='./dataset/train.csv'
    train_labels=[]
    train_pixels=[]

    with open(train_file_path, 'r') as f:
        # for i, data in enumerate(f):
        #     print("i {}, data {}".format(i, data))
        #     csv_data = data.split(',')
        #
        #     print("csv_data {}".format(csv_data))
        next(f)
        reader = csv.reader(f)
        for row in reader:
            # print("row {}".format(row[1]))
            train_pixels.append(np.fromstring(row[1], dtype=float, sep=' ').reshape((48, 48, 1)))

            onehot = np.zeros((7, ), dtype=np.float)
            onehot[int(row[0])] = 1.

            train_labels.append(onehot)

            train_pixels_array = np.array(train_pixels)
            train_labels_array = np.array(train_labels)

            if len(train_labels_array) > 10000:
                break

    print("train pixel shape {}".format(train_pixels_array.shape))
    print("train label shape {}".format(train_labels_array.shape))

    return train_pixels_array, train_labels_array


def load_test_data():
    test_file_path = './dataset/test.csv'
    test_pixels = []

    with open(test_file_path, 'r') as f:
        next(f)
        reader = csv.reader(f)
        for row in reader:
            test_pixels.append(np.fromstring(row[1], dtype=float, sep=' ').reshape((48, 48, 1)))
            test_pixels_array = np.array(test_pixels)

    print("test pixel shape {}".format(test_pixels_array.shape))

    return test_pixels_array

def shuffle(pixels, labels):
    p = np.random.permutation(len(pixels))
    return pixels[p], labels[p]

def separate(pixels, labels, test_num):
    train_num = len(pixels) - test_num
    return pixels[:train_num], labels[:train_num], pixels[train_num:], labels[train_num:]

if __name__=='__main__':
    load_train_data()
    load_test_data()
        # reader = csv.reader(f)
        # for row in reader:
        #     train_labels.append(row[0])
        #     train_pixels.append(row[1].split(' '))
        #
        # for i in range(len(valid_labels)):


    # with open(test_file_path, 'r') as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         test_labels.append(row[0])
    #         test_pixels.append(row[1].split(' '))
    #
    # return train_pixels, train_labels, test_pixels, test_labels