import os
import random
import shutil


def generate_training_and_validation_sets(training_percentage=0.5):

    current_directory = '/home/jlazo/MI_BIBLIOTECA/Datasets/BUS_project/all_train_och_valid/'
    files_path_positives = "".join([current_directory, 'benign/'])
    files_path_negatives = "".join([current_directory, 'malignant/'])
    positive_images = os.listdir(files_path_positives)
    negative_images = os.listdir(files_path_negatives)

    training_dir = '/home/jlazo/MI_BIBLIOTECA/Datasets/BUS_project/training_validation/training/'
    validation_dir = '/home/jlazo/MI_BIBLIOTECA/Datasets/BUS_project/training_validation/validation/'
    
    #all_training = '/home/william/m18_jorge/Desktop/THESIS/DATA/transfer_learning_training/all_training/'
    #all_validation = '/home/william/m18_jorge/Desktop/THESIS/DATA/transfer_learning_training/all_validation/'

    for count_i, image in enumerate(positive_images):
        print(count_i,len(positive_images))
        if random.random() <= training_percentage:
            shutil.copy(files_path_positives + image, "".join([training_dir, 'benign/', image]))
            #shutil.copy(files_path_positives + image, "".join([all_training, image]))
        else:
            shutil.copy(files_path_positives + image, "".join([validation_dir, 'benign/', image]))
            #shutil.copy(files_path_positives + image, "".join([all_validation, image]))
            
    for count_i, image in enumerate(negative_images):
        print(count_i,len(negative_images))
        if random.random() <= training_percentage:
            shutil.copy(files_path_negatives + image, "".join([training_dir, 'malignant/', image]))
            #shutil.copy(files_path_negatives + image, "".join([all_training, image]))
        else:
            shutil.copy(files_path_negatives + image, "".join([validation_dir, 'malignant/', image]))
            #shutil.copy(files_path_negatives + image, "".join([all_validation, image]))


def main():
    generate_training_and_validation_sets(training_percentage=0.6)


if __name__ == '__main__':
    main()
