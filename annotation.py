import os
import random
import numpy as np


def write_each_txt(file_list, txt):

    file_txt = open("./model_data/" + txt, "w")
    wd = os.getcwd()

    for name in file_list:

        obj1 = name.split(".")
        obj2 = obj1[0]
        obj3 = obj2.split("_")

        category = str(0)
        information = " " + obj3[1] + "," + obj3[2] + "," + obj3[3] + "," + obj3[4] + "," + category

        file_txt.write(wd + "/crop_ccpd/" + name + information + "\n")
    file_txt.close()


def write_txt():

    filename = os.listdir('crop_ccpd')
    filename.sort()

    index = list(np.arange(0, len(filename), 1))
    random.seed(10)
    random.shuffle(index)
    filename = [filename[k] for k in index]

    train_file = filename[:2900]
    val_file = filename[2900:3000]
    test_file = filename[3000:]

    write_each_txt(train_file, '2021_train.txt')
    write_each_txt(val_file, '2021_val.txt')
    write_each_txt(test_file, '2021_test.txt')


write_txt()
