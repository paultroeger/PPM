"""
PyTorch PATE pipeline runner.

Trains the full teacher ensemble, then the student, then reports the GNMax
epsilon.

Example:
    python project/pate_pytorch/pate.py --dataset=svhn --nb_teachers=10 \
        --teacher_epochs=30 --student_epochs=80 --student_queries=1000 \
        --noise_scale=20.0 --delta=1e-5
"""

import argparse
import os

from train_teachers import train_teacher
from train_student import train_student


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mnist", "svhn", "cifar10"),
                        required=True)
    parser.add_argument("--nb_teachers", type=int, default=10)
    parser.add_argument("--teacher_epochs", type=int, default=10)
    parser.add_argument("--student_epochs", type=int, default=10)
    parser.add_argument("--student_queries", type=int, default=1000)
    parser.add_argument("--noise_scale", type=float, default=10.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--train_dir", default="/tmp/pate_pytorch_train")
    args = parser.parse_args()

    os.makedirs(args.train_dir, exist_ok=True)

    # train each teacher on its disjoint partition of the training data
    for teacher_id in range(args.nb_teachers):
        print("Training teacher " + str(teacher_id) + "/" +
              str(args.nb_teachers))
        train_teacher(args.dataset, args.nb_teachers, teacher_id,
                      epochs=args.teacher_epochs,
                      batch_size=args.batch_size, lr=args.lr,
                      data_dir=args.data_dir,
                      train_dir=args.train_dir)

    metrics_file = os.path.join(
        args.train_dir,
        "{}_{}_teachers_gnmax_sigma_{}_metrics.json".format(
            args.dataset, args.nb_teachers, args.noise_scale))

    # train the student on noisily aggregated teacher votes and run the
    # GNMax accountant to get the privacy cost
    train_student(args.dataset, args.nb_teachers,
                  stdnt_share=args.student_queries,
                  noise_scale=args.noise_scale, delta=args.delta,
                  epochs=args.student_epochs,
                  batch_size=args.batch_size, lr=args.lr,
                  data_dir=args.data_dir, train_dir=args.train_dir,
                  teachers_dir=args.train_dir,
                  metrics_file=metrics_file, save_labels=True)

    print("Metrics written to " + metrics_file)


if __name__ == "__main__":
    main()
