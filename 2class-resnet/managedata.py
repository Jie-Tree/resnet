import os
from PIL import Image
import numpy as np
import xlrd
from sklearn.model_selection import train_test_split

def read_image(img_name):
    im = Image.open(img_name)
    # im = Image.open(img_name).convert('1')
    # im.show()
    data = np.array(im)
    # print(data)
    # print(data.shape)
    data = np.reshape(data, [512, 512, 1])
    # data = data[:, :, np.newaxis]
    # print(data.shape)
    # print(data[0][69])
    # exit()
    return data


def load_img_data(img_path):
    images = []
    # if not os.path.isdir(file):
    #     return
    file_list = os.listdir(img_path)
    file_list.sort()
    for file in file_list:
        # print(file)
        images.append(read_image(img_path + '/' + file))
    print('load img success!')
    X = np.array(images)
    # print(X.shape)
    return X


def num2lable(number):
    if number == 0:
        Grading = 'Low'
        Staging = 'MIBC'
    if number == 1:
        Grading = 'High'
        Staging = 'MIBC'
    if number == 2:
        Grading = 'Low'
        Staging = 'NMIBC'
    if number == 3:
        Grading = 'High'
        Staging = 'NMIBC'
    return Grading, Staging

def load_label_data(label_path):
    labels = []
    label_sheet = xlrd.open_workbook(label_path + '/DataInfo.xlsx').sheets()[0]
    for i in range(1, label_sheet.nrows):
        # for i in range(1, 1000):
        ImageName, Grading, Staging = label_sheet.row_values(i)
        if Grading == '\'Low\'' and Staging == '\'MIBC\'':
            labels.append(0)
        elif Grading == '\'Low\'' and Staging == '\'NMIBC\'':
            labels.append(1)
        elif Grading == '\'High\'' and Staging == '\'MIBC\'':
            labels.append(2)
        elif Grading == '\'High\'' and Staging == '\'NMIBC\'':
            labels.append(3)
    return np.array(labels)


def load_grading_label_data(label_path):
    labels = []
    label_sheet = xlrd.open_workbook(label_path + '/DataInfo.xlsx').sheets()[0]
    for i in range(1, label_sheet.nrows):
        # for i in range(1, 1000):
        ImageName, Grading, Staging = label_sheet.row_values(i)
        if Grading == '\'Low\'':
            labels.append(0)
        elif Grading == '\'High\'':
            labels.append(1)
    return np.array(labels)


def load_staging_label_data(label_path):
    labels = []
    label_sheet = xlrd.open_workbook(label_path + '/DataInfo.xlsx').sheets()[0]
    for i in range(1, label_sheet.nrows):
        # for i in range(1, 1000):
        ImageName, Grading, Staging = label_sheet.row_values(i)
        if Staging == '\'MIBC\'':
            labels.append(0)
        elif Staging == '\'NMIBC\'':
            labels.append(1)
    return np.array(labels)

def load_data():
    X = load_img_data('TrainingData')
    print('success load img')
    print(X.shape)
    Y = load_label_data('TrainingLabel')
    print('success load label')
    print(Y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    return (X_train, y_train), (X_test, y_test)


def load_grading_data():
    X = load_img_data('TrainingData')
    print('success load img')
    print(X.shape)
    Y = load_grading_label_data('TrainingLabel')
    print('success load label')
    print(Y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    return (X_train, y_train), (X_test, y_test)


def load_staging_data():
    X = load_img_data('TrainingData')
    print('success load img')
    print(X.shape)
    Y = load_staging_label_data('TrainingLabel')
    print('success load label')
    print(Y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    return (X_train, y_train), (X_test, y_test)

'''
Low = 0
High = 1
MIBC = 0
NMIBC = 1
'''
def load_2class_data():
    X = load_img_data('TrainingData')
    print('success load img')
    print(X.shape)
    Y = load_label_data('TrainingLabel')
    print('success load label')
    print(Y.shape)
    y_train_grading =[]
    y_train_staging = []
    y_test_grading = []
    y_test_staging = []
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    for trn in y_train:
        if trn == 0:
            y_train_grading.append(0)
            y_train_staging.append(0)
        
        if trn == 1:
            y_train_grading.append(0)
            y_train_staging.append(1)
        
        if trn == 2:
            y_train_grading.append(1)
            y_train_staging.append(0)
        
        if trn == 3:
            y_train_grading.append(1)
            y_train_staging.append(1)

    for ten in y_test:
        if ten == 0:
            y_test_grading.append(0)
            y_test_staging.append(0)

        if ten == 1:
            y_test_grading.append(0)
            y_test_staging.append(1)

        if ten == 2:
            y_test_grading.append(1)
            y_test_staging.append(0)

        if ten == 3:
            y_test_grading.append(1)
            y_test_staging.append(1)

    return (np.array(X_train), np.array(y_train_grading), np.array(y_train_staging)), \
           (np.array(X_test), np.array(y_test_grading), np.array(y_test_staging))


if __name__ == '__main__':
    (X_train, y_train_grading, y_train_staging), (X_test, y_test_grading, y_test_staging) = load_2class_data()
    print('done')

