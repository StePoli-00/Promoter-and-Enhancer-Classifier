#codice utilizzato per splittare il dataset in Training, Validation e Testing in rapporto 80/10/10



import os
import shutil
import random 
import argparse


training_path=""
validation_path=""
testing_path=""



if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Script to split Dataset into training, validation and testing.')
    parser.add_argument('-train',"--training_path", help='path to the training Dataset folder')
    parser.add_argument('-test', '--testing_path', help='path to the testing Dataset folder')
    parser.add_argument('-validation',"--validation_path",help="path to the validation Dataset folder")
    args=parser.parse_args()
    
    training_path=args.training_path
    validation_path=args.testing_path
    testing_path=args.validation_path

    if training_path is None or testing_path is None or validation_path is None:
       parser.print_help()
       parser.error("Parameters error")


    training_ids_path = os.path.join(training_path, "ids")
    training_att_mask_path = os.path.join(training_path, "att_mask")
    training_labels_path = os.path.join(training_path, "labels")

    validation_ids_path = os.path.join(validation_path, "ids")
    validation_att_mask_path = os.path.join(validation_path, "att_mask")
    validation_labels_path = os.path.join(validation_path, "labels")

    testing_ids_path = os.path.join(testing_path, "ids")
    testing_att_mask_path = os.path.join(testing_path, "att_mask")
    testing_labels_path = os.path.join(testing_path, "labels")



    for el in os.listdir(validation_ids_path):
        shutil.move(os.path.join(validation_ids_path, el), training_ids_path)
        shutil.move(os.path.join(validation_att_mask_path, el), training_att_mask_path)
        shutil.move(os.path.join(validation_labels_path, el), training_labels_path)

    for el in os.listdir(testing_ids_path):
        shutil.move(os.path.join(testing_ids_path, el), training_ids_path)
        shutil.move(os.path.join(testing_att_mask_path, el), training_att_mask_path)
        shutil.move(os.path.join(testing_labels_path, el), training_labels_path)

    num_elem = len(os.listdir(training_ids_path))

    for el in os.listdir(training_ids_path):
        i = random.randint(0,num_elem-1)
        if i % 10 == 0:
            shutil.move(os.path.join(training_ids_path, el), validation_ids_path)
            shutil.move(os.path.join(training_att_mask_path, el), validation_att_mask_path)
            shutil.move(os.path.join(training_labels_path, el), validation_labels_path)
        if i % 10 == 1:
            shutil.move(os.path.join(training_ids_path, el), testing_ids_path)
            shutil.move(os.path.join(training_att_mask_path, el), testing_att_mask_path)
            shutil.move(os.path.join(training_labels_path, el), testing_labels_path)

print("Dataset correctly splitted in Training, Testing and Validation set in these corresponding folders : "
      +"\n"+training_path
      +"\n"+validation_path
      +"\n"+testing_path
      )





    