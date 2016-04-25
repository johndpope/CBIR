import os
import random

class_types = ['face', 'flower']
input_ext = '.dat'
dataset_folder = '.'
train_dataset_size = 3000
libsvm_dir = './libsvm-3.21'
train_command = libsvm_dir + '/svm-train -c 8 -g 0.125'
predict_command = libsvm_dir + '/svm-predict'
subset_command = 'python ' + libsvm_dir + '/tools/subset.py'
data_root_folder = '.'

for ctype in class_types:
    print 'Splitting ' + ctype + '...'
    dataset_file = os.path.join(data_root_folder, ctype + input_ext)
    tr_subset_file = os.path.join(data_root_folder, ctype + '.tr')
    te_subset_file = os.path.join(data_root_folder, ctype + '.te')
    print subset_command + ' ' + dataset_file + ' ' + str(train_dataset_size) + ' ' + tr_subset_file + ' ' + te_subset_file
    os.system(subset_command + ' ' + dataset_file + ' ' + str(train_dataset_size) + ' ' + tr_subset_file + ' ' + te_subset_file)

for ctype in class_types:
    print 'Training ' + ctype + '...'
    tr_set_file = os.path.join(data_root_folder, ctype + '.tr')
    model_file = os.path.join(data_root_folder, ctype + '.model')
    print train_command + ' ' + tr_set_file + ' ' + model_file
    os.system(train_command + ' ' + tr_set_file + ' ' + model_file)

for ctype in class_types:
    print 'Accuracy for ' + ctype + '...'
    te_set_file = os.path.join(data_root_folder, ctype + '.te')
    model_file = os.path.join(data_root_folder, ctype + '.model')
    output_temp_file = os.path.join(data_root_folder, ctype + '.temp')
    os.system(predict_command + ' ' + te_set_file + ' ' + model_file + ' ' + output_temp_file)