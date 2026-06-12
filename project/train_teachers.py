"""
PyTorch port of pate_2017/train_teachers.py.

Trains one teacher out of an ensemble of nb_teachers on its own disjoint
partition of the training data. The TF1 flags became argparse arguments and
max_steps became epochs, otherwise the structure is the same.
"""

import argparse
import os

import deep_cnn
import metrics
import test_load_data


def train_teacher(dataset, nb_teachers, teacher_id, epochs=10, batch_size=64,
                  lr=0.01, data_dir='./data', train_dir='/tmp/train_dir'):
    """This function trains a teacher (teacher id) among an ensemble of nb_teachers
    models for the dataset specified."""
    os.makedirs(train_dir, exist_ok=True)

    # load the dataset as tensors
    if dataset == 'mnist':
        train_data, train_labels, test_data, test_labels = test_load_data.ld_mnist(data_dir=data_dir)
    elif dataset == 'svhn':
        train_data, train_labels, test_data, test_labels = test_load_data.ld_svhn(data_dir=data_dir)
    elif dataset == 'cifar10':
        train_data, train_labels, test_data, test_labels = test_load_data.ld_cifar10(data_dir=data_dir)
    else:
        print("Check value of dataset flag")
        return False

    # retrieve subset of data for this teacher
    data, labels = test_load_data.partition_dataset(train_data,
                                                    train_labels,
                                                    nb_teachers,
                                                    teacher_id)

    print("Length of training data: " + str(len(labels)))

    # define teacher checkpoint filename and full path
    filename = str(nb_teachers) + '_teachers_' + str(teacher_id) + '.pt'
    ckpt_path = train_dir + '/' + str(dataset) + '_' + filename

    # perform teacher training
    deep_cnn.train(dataset, data, labels, ckpt_path,
                   epochs=epochs, batch_size=batch_size, lr=lr)

    # retrieve teacher probability estimates on the test data
    teacher_preds = deep_cnn.softmax_preds(dataset, test_data, ckpt_path)

    # compute teacher accuracy
    precision = metrics.accuracy(teacher_preds, test_labels.numpy())
    print('Precision of teacher after training: ' + str(precision))

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('mnist', 'svhn', 'cifar10'),
                        default='svhn', help='The name of the dataset to use')
    parser.add_argument('--nb_teachers', type=int, default=50,
                        help='Teachers in the ensemble.')
    parser.add_argument('--teacher_id', type=int, default=0,
                        help='ID of teacher being trained.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs to run.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01,
                        help='SGD learning rate')
    parser.add_argument('--data_dir', default='./data',
                        help='Where torchvision downloads the datasets')
    parser.add_argument('--train_dir', default='/tmp/train_dir',
                        help='Where model checkpoints are saved')
    args = parser.parse_args()

    train_teacher(args.dataset, args.nb_teachers, args.teacher_id,
                  epochs=args.epochs, batch_size=args.batch_size,
                  lr=args.lr, data_dir=args.data_dir,
                  train_dir=args.train_dir)


if __name__ == '__main__':
    main()
