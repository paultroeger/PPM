import argparse
import os
import subprocess
import sys


def _run(cmd):
    print("Running: " + " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mnist", "svhn", "cifar10"),
                        required=True)
    parser.add_argument("--nb_teachers", type=int, default=10)
    parser.add_argument("--teacher_steps", type=int, default=3000)
    parser.add_argument("--student_steps", type=int, default=3000)
    parser.add_argument("--student_queries", type=int, default=1000)
    parser.add_argument("--noise_scale", type=float, default=10.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--deeper", action="store_true")
    args = parser.parse_args()

    batch_size = 128
    data_dir = "/tmp/pate_data"
    train_dir = "/tmp/pate_train"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_teachers = os.path.join(script_dir, "pate_2017", "train_teachers.py")
    train_student = os.path.join(script_dir, "pate_2017", "train_student.py")

    for teacher_id in range(args.nb_teachers):
        cmd = [
            sys.executable,
            train_teachers,
            "--dataset=" + args.dataset,
            "--nb_teachers=" + str(args.nb_teachers),
            "--teacher_id=" + str(teacher_id),
            "--max_steps=" + str(args.teacher_steps),
            "--batch_size=" + str(batch_size),
            "--data_dir=" + data_dir,
            "--train_dir=" + train_dir,
        ]
        if args.deeper:
            cmd.append("--deeper")
        _run(cmd)

    metrics_file = os.path.join(
        train_dir,
        "{}_{}_teachers_gnmax_sigma_{}_metrics.json".format(
            args.dataset, args.nb_teachers, args.noise_scale))

    cmd = [
        sys.executable,
        train_student,
        "--dataset=" + args.dataset,
        "--nb_teachers=" + str(args.nb_teachers),
        "--teachers_max_steps=" + str(args.teacher_steps),
        "--max_steps=" + str(args.student_steps),
        "--stdnt_share=" + str(args.student_queries),
        "--noise_scale=" + str(args.noise_scale),
        "--delta=" + str(args.delta),
        "--batch_size=" + str(batch_size),
        "--data_dir=" + data_dir,
        "--train_dir=" + train_dir,
        "--teachers_dir=" + train_dir,
        "--metrics_file=" + metrics_file,
    ]
    if args.deeper:
        cmd.append("--deeper")
    _run(cmd)
    print("Metrics written to " + metrics_file)


if __name__ == "__main__":
    main()

# python project/pate.py --dataset=cifar10 --nb_teachers=30 --teacher_steps=1000 --student_steps=2000 --student_queries=1000 --noise_scale=10.0 --delta=1e-5