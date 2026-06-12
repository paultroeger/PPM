"""
PyTorch port of pate_2017/train_student.py.

The student is trained on a slice of the test set whose labels come from the
noisy (GNMax) aggregation of the teacher ensemble's votes.
"""

import argparse
import json
import os

import numpy as np
import torch

import aggregation
import deep_cnn
import gnmax_accountant
import metrics
import test_load_data


def ensemble_preds(dataset, nb_teachers, stdnt_data, teachers_dir):
    """Given a dataset, a number of teachers, and some input data, this helper
    function queries each teacher for predictions on the data and returns
    all predictions in a single array. That can then be aggregated into
    one single prediction per input using aggregation.py."""
    # compute shape of array that will hold probabilities produced by each
    # teacher, for each training point, and each output class
    result_shape = (nb_teachers, len(stdnt_data), 10)

    # create array that will hold result
    result = np.zeros(result_shape, dtype=np.float32)

    # get predictions from each teacher
    for teacher_id in range(nb_teachers):
        # compute path of checkpoint file for teacher model with ID teacher_id
        ckpt_path = teachers_dir + '/' + str(dataset) + '_' + str(nb_teachers) + '_teachers_' + str(teacher_id) + '.pt'

        # get predictions on our training data and store in result array
        result[teacher_id] = deep_cnn.softmax_preds(dataset, stdnt_data, ckpt_path)

        # this can take a while when there are a lot of teachers so output status
        print("Computed Teacher " + str(teacher_id) + " softmax predictions")

    return result


def prepare_student_data(dataset, nb_teachers, stdnt_share, noise_scale,
                         data_dir='./data', teachers_dir='/tmp/train_dir',
                         save=False):
    """Takes a dataset name and the size of the teacher ensemble and prepares
    training data for the student model."""
    # load the test split only; the student never touches the training set
    if dataset == 'mnist':
        test_data, test_labels = test_load_data.ld_mnist(test_only=True, data_dir=data_dir)
    elif dataset == 'svhn':
        test_data, test_labels = test_load_data.ld_svhn(test_only=True, data_dir=data_dir)
    elif dataset == 'cifar10':
        test_data, test_labels = test_load_data.ld_cifar10(test_only=True, data_dir=data_dir)
    else:
        print("Check value of dataset flag")
        return False

    # make sure there is data leftover to be used as a test set
    assert stdnt_share < len(test_data)

    # prepare [unlabeled] student training data (subset of test set)
    stdnt_data = test_data[:stdnt_share]

    # compute teacher predictions for student training data
    teachers_preds = ensemble_preds(dataset, nb_teachers, stdnt_data,
                                    teachers_dir)

    # aggregate teacher predictions to get student training labels.
    # we always request the clean vote histograms because the GNMax privacy
    # accounting consumes them (the TF version only did so when saving).
    stdnt_labels, clean_votes, labels_for_dump = aggregation.noisy_max(
        teachers_preds, noise_scale, return_clean_votes=True)

    if save:
        # name exported files by GNMax sigma to avoid mixing different runs
        filepath = data_dir + "/" + str(dataset) + '_' + str(nb_teachers) + '_student_clean_votes_gnmax_sigma_' + str(
            noise_scale) + '.npy'
        filepath_labels = data_dir + "/" + str(dataset) + '_' + str(
            nb_teachers) + '_teachers_labels_gnmax_sigma_' + str(noise_scale) + '.npy'
        filepath_stdnt = data_dir + "/" + str(dataset) + '_' + str(nb_teachers) + '_student_labels_gnmax_sigma_' + str(
            noise_scale) + '.npy'

        with open(filepath, 'wb') as file_obj:
            np.save(file_obj, clean_votes)
        with open(filepath_labels, 'wb') as file_obj:
            np.save(file_obj, labels_for_dump)
        with open(filepath_stdnt, 'wb') as file_obj:
            np.save(file_obj, stdnt_labels)

    # print accuracy of aggregated labels
    ac_ag_labels = metrics.accuracy(stdnt_labels,
                                    test_labels[:stdnt_share].numpy())
    print("Accuracy of the aggregated labels: " + str(ac_ag_labels))

    # store unused part of test set for use as a test set after student training
    stdnt_test_data = test_data[stdnt_share:]
    stdnt_test_labels = test_labels[stdnt_share:]

    return stdnt_data, stdnt_labels, stdnt_test_data, stdnt_test_labels, clean_votes


def train_student(dataset, nb_teachers, stdnt_share=1000, noise_scale=10.0,
                  delta=1e-5, epochs=10, batch_size=64, lr=0.01,
                  data_dir='./data', train_dir='/tmp/train_dir',
                  teachers_dir='/tmp/train_dir', metrics_file='',
                  save_labels=False):
    """This function trains a student using predictions made by an ensemble of
    teachers. The student and teacher models are trained using the same CNN."""
    os.makedirs(train_dir, exist_ok=True)

    # call helper function to prepare student data using teacher predictions
    stdnt_data, stdnt_labels, stdnt_test_data, stdnt_test_labels, clean_votes = \
        prepare_student_data(dataset, nb_teachers, stdnt_share, noise_scale,
                             data_dir=data_dir, teachers_dir=teachers_dir,
                             save=save_labels)

    # prepare checkpoint filename and path
    ckpt_path = train_dir + '/' + str(dataset) + '_' + str(nb_teachers) + '_student.pt'

    # the aggregation returns numpy int32 labels; deep_cnn.train expects tensors
    stdnt_label_tensor = torch.from_numpy(np.asarray(stdnt_labels, dtype=np.int64))

    # start student training
    deep_cnn.train(dataset, stdnt_data, stdnt_label_tensor, ckpt_path,
                   epochs=epochs, batch_size=batch_size, lr=lr)

    # compute student label predictions on remaining chunk of test set
    student_preds = deep_cnn.softmax_preds(dataset, stdnt_test_data, ckpt_path)

    # compute student accuracy
    precision = metrics.accuracy(student_preds, stdnt_test_labels.numpy())
    print('Precision of student after training: ' + str(precision))

    # GNMax RDP accountant from the 2018 codebase (framework independent)
    privacy_result = gnmax_accountant.compute_epsilon(
        clean_votes,
        sigma=noise_scale,
        delta=delta,
        include_smooth_sensitivity=True)
    print('GNMax epsilon at delta ' + str(delta) + ': ' +
          str(privacy_result['epsilon']))
    print('Optimal Renyi order: ' + str(privacy_result['optimal_order']))

    if metrics_file:
        # machine-readable output for the cross-method privacy/utility table
        metrics_payload = {
            'dataset': dataset,
            'nb_teachers': nb_teachers,
            'student_queries': int(stdnt_share),
            'noise_scale': float(noise_scale),
            'delta': float(delta),
            'epsilon': privacy_result['epsilon'],
            'optimal_order': privacy_result['optimal_order'],
            'smooth_sensitivity': True,
            'student_accuracy': precision,
        }
        with open(metrics_file, 'w') as file_obj:
            json.dump(metrics_payload, file_obj, indent=2, sort_keys=True)

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('mnist', 'svhn', 'cifar10'),
                        default='svhn', help='The name of the dataset to use')
    parser.add_argument('--nb_teachers', type=int, default=10,
                        help='Teachers in the ensemble.')
    parser.add_argument('--stdnt_share', type=int, default=1000,
                        help='Student share (last index) of the test data')
    parser.add_argument('--noise_scale', type=float, default=10.0,
                        help='Standard deviation of the Gaussian noise for GNMax')
    parser.add_argument('--delta', type=float, default=1e-5,
                        help='Target delta for the GNMax privacy accountant')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to run student.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01,
                        help='SGD learning rate')
    parser.add_argument('--data_dir', default='./data',
                        help='Where torchvision downloads the datasets')
    parser.add_argument('--train_dir', default='/tmp/train_dir',
                        help='Where the student checkpoint is saved')
    parser.add_argument('--teachers_dir', default='/tmp/train_dir',
                        help='Directory where teachers checkpoints are stored.')
    parser.add_argument('--metrics_file', default='',
                        help='Optional JSON file for student accuracy and epsilon')
    parser.add_argument('--save_labels', action='store_true',
                        help='Dump numpy arrays of labels and clean teacher votes')
    args = parser.parse_args()

    train_student(args.dataset, args.nb_teachers,
                  stdnt_share=args.stdnt_share,
                  noise_scale=args.noise_scale, delta=args.delta,
                  epochs=args.epochs, batch_size=args.batch_size,
                  lr=args.lr, data_dir=args.data_dir,
                  train_dir=args.train_dir,
                  teachers_dir=args.teachers_dir,
                  metrics_file=args.metrics_file,
                  save_labels=args.save_labels)


if __name__ == '__main__':
    main()
