import csv
import cv2
import numpy as np
import os
from scipy import misc
from skimage import transform
import shutil


def load_predictions(csv_file):
    labels = []
    image_name = []
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            labels.append(row[1])

            image_name.append(row[2])
    return labels, image_name


def load_labels(csv_file):
    labels = []
    image_name = []
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            labels.append(float(row[1]))
            image_name.append(row[2])
    return labels, image_name


def make_csv_real_values(path, name_output):

    subfolders = os.listdir(path)
    for folder in subfolders:
        new_path = ''.join([path, folder, '/'])
        if folder == 'positives':
            files_1 = os.listdir(new_path)
        elif folder == 'negatives':
            files_2 = os.listdir(new_path)

    with open(''.join([name_output, '.csv']), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for i, row in enumerate(files_1):
            writer.writerow([row, 0])

        for i, row in enumerate(files_2):
            writer.writerow([row, 1])


def generate_ensemble_results(directory, file_name_1, directory2, file_name_2):

    ensemble_values = []
    matched_names = []

    y1, names_1 = load_predictions(directory+file_name_1)
    y2, names_2 = load_predictions(directory2+file_name_2)

    for k, name in enumerate(names_1[:]):
        for j, other_name in enumerate(names_2[:]):
            if name != 'fname':
                if name != 'Name':
                    if name[15:] == other_name[15:]:
                        matched_names.append(name)
                        ensemble_values.append(0.5*float(y1[k]) + 0.5*float(y2[j]))

    return ensemble_values, matched_names


def match_reals_and_prediction(file_reals, files_predictions, name_ouput):
    list_reals = []
    list_predictons = []
    common_names = []
    reals, name_reals = load_labels(file_reals)
    predictions, names_predictions = load_predictions(files_predictions)
    for i, name in enumerate(name_reals):
        for j, other_name in enumerate(names_predictions):
            if name == other_name:
                common_names.append(name)
                list_reals.append(float(reals[i]))
                list_predictons.append(float(predictions[j]))
                break

    with open(''.join([name_ouput, '.csv']), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Name Picture', 'Real Value', 'Predicted Value'])
        for i, row in enumerate(common_names):
            writer.writerow([row, list_reals[i], list_predictons[i]])


def load_pictures_1(directory):
    directory = directory
    lista = [f for f in os.listdir(directory)]
    imgs = np.zeros([len(lista), 100, 100, 3])

    for i, image in enumerate(lista):
        img = misc.imread(''.join([directory, image]))
        if np.array_equal(np.shape(img), (100, 100, 3)):
            imgs[i] = img
        else:
            img = transform.resize(img, (100, 100, 3))
            imgs[i] = img

    array = np.array(imgs)
    array.reshape(len(imgs), 100, 100, 3)
    # return np.array(imgs[:])
    return array, lista


def load_images_with_labels(directory):

    sub_folders = os. listdir(directory)
    all = []
    total = 0
    print(''.join([str(len(sub_folders)), 'classes found: ', '']))

    for folder in sub_folders:
        files = os.listdir(''.join([directory, '/', folder]))
        print(''.join([folder, ' ', str(len(files)), ' files']))
        total = total + len(files)
        all.append(files)
    print('total images: ', total)
    print(len(all))
    return all


def load_history(csv_file):
    acc = []
    val_acc = []
    loss = []
    val_loss = []
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if row[0]!= 'Acc':
                acc.append(float(row[0]))
                val_acc.append(float(row[1]))
                loss.append(float(row[2]))
                val_loss.append(float(row[3]))

    return acc, val_acc, loss, val_loss


def load_images_with_labels(directory):

    sub_folders = os. listdir(directory)
    all = []
    total = 0
    print(''.join([str(len(sub_folders)), ' classes found: ', '']))

    for folder in sub_folders:
        files = os.listdir(''.join([directory, '/', folder]))
        print(''.join([folder, ' ', str(len(files)), ' files']))
        total = total + len(files)
        all.append(files)
    print('total images: ', total)

    return sub_folders, all


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def generate_k_sets(directory, destination, k):
    base = destination
    # check sub-folders and images
    subfolders, lista = load_images_with_labels(directory)

    # determine how many images there are on each folder and the rate of it
    number_each_class = [len(i) for i in lista]
    rate = (min(number_each_class) / max(number_each_class))

    # create k sub-folders
    for number in range(k):
        ensure_dir(''.join([destination, '/k_', str(number)]))

    while lista[0] or lista[1]:
        if np.random.rand() > rate:
            folder_choice = number_each_class.index(max(number_each_class))
        else:
            folder_choice = number_each_class.index(min(number_each_class))

        if lista[folder_choice]:
            image = np.random.choice(lista[folder_choice])
            lista[folder_choice].remove(image)
            img = ''.join([directory, subfolders[folder_choice], '/', image])
            k_choice = np.random.randint(0, 4)
            destination = ''.join([base, 'k_', str(k_choice), '/', subfolders[folder_choice], '/', image])
            ensure_dir(''.join([base, 'k_', str(k_choice), '/', subfolders[folder_choice], '/']))
            print(destination)
            shutil.copyfile(img, destination)


def create_training_and_validation_list(directory, percentage_training):
    labels_training = []
    images_training = []
    labels_validation = []
    images_validation = []
    subfolders, lista = load_images_with_labels(directory)
    k = [len(i) for i in lista]
    rate = (min(k)/max(k))
    while lista[0] or lista[1]:

        if np.random.rand() > rate:
            if lista[k.index(max(k))]:
                image = np.random.choice(lista[k.index(max(k))])
                lista[k.index(max(k))].remove(image)
                choice = 1
                img = ''.join([directory, subfolders[k.index(max(k))], '/', image])
        else:
            if lista[k.index(min(k))]:
                image = np.random.choice(lista[k.index(min(k))])
                lista[k.index(min(k))].remove(image)
                choice = 0
                img = ''.join([directory, subfolders[k.index(min(k))], '/', image])

        if np.random.rand() > percentage_training:
            print('load: ', image, choice)
            labels_validation.append([image, choice])
            images_validation.append(cv2.imread(img))
        else:
            labels_training.append([image, choice])
            images_training.append(cv2.imread(img))

    return labels_validation, images_validation, labels_training, images_training


def paint_image(image, value, idx=None):

    im = cv2.imread(image)
    real_ones = [86, 87, 88, 67, 68, 147, 148, 128, 108, 72, 73, 74, 54]

    """if idx is not None:
        if idx in real_ones:
            if value > 0.9:
                im[:, :, 1] = 100
            else:
                im[:, :, 2] = 100

        else:
            if value > 0.9:
                im[:, :, 0] = 150"""


    #if value > 0.5:
    #    im[:, :, 2] = 50

    # ---rgb----

    if 0.1 < value < 0.3:
        im[:, :, 0] = 50
    elif 0.3 < value < 0.5:
        im[:, :, 0] = 100
    elif value > 0.5:
        im[:, :, 0] = 90

    """"# --- IR -----

    if 0.1 < value < 0.3:
        im[:, :, 0] = 50
    elif 0.3 < value < 0.5:
        im[:, :, 2] = 50
    elif value > 0.5:
        im[:, :, 2] = 25"""

    return im


