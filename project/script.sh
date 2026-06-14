#!/usr/bin/env bash
# =============================================================================
# PATE Experiment Batch Runner
# Trains each teacher ensemble once, then sweeps noise_scale for the student.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Fixed hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE=128
STUDENT_EPOCHS=80
DELTA=1e-5
LEARNING_RATE=0.01
STUDENT_QUERIES=1000
DATA_DIR="./data"
BASE_TRAIN_DIR="/tmp/pate_pytorch_train"

# ---------------------------------------------------------------------------
# Variable hyperparameters
# ---------------------------------------------------------------------------
TEACHER_EPOCHS_LIST=(10 50 100)

# ---------------------------------------------------------------------------
# Log directory
# ---------------------------------------------------------------------------
LOG_DIR="logs/pate_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.tsv"

printf "phase\tdataset\tnb_teachers\tteacher_epochs\tnoise_scale\tepsilon_target\twall_time_s\tstatus\tlog_file\n" \
    > "$SUMMARY"

# ---------------------------------------------------------------------------
# Parameter grid
# Format:  <nb_teachers>: "<eps1_noise> <eps4_noise> <eps8_noise>"
# ---------------------------------------------------------------------------
declare -A NOISE_SCALES=(
    # nb_teachers   eps=1    eps=4    eps=8
    #["10"]="150.0   40.0   20.0"
    #["50"]="80.0    22.0   11.0"
    #["100"]="55.0   15.0    8.0"
    ["100"]="200 57 6.7"
)

EPSILON_TARGETS=(1 4 8)

DATASETS=("mnist" "svhn" "cifar10")
NB_TEACHERS=(100)

# ---------------------------------------------------------------------------
# Train all teachers for a (dataset, nb_teachers, teacher_epochs) combo.
# ---------------------------------------------------------------------------
train_teachers() {
    local dataset="$1"
    local nb_teachers="$2"
    local teacher_epochs="$3"
    local train_dir="$4"

    echo "===================================================================="
    echo "  Training $nb_teachers teachers  |  dataset=$dataset  epochs=$teacher_epochs"
    echo "  Checkpoints -> $train_dir"
    echo "===================================================================="

    local log_file="$LOG_DIR/teachers_${dataset}_t${nb_teachers}_e${teacher_epochs}.log"
    local start
    start=$(date +%s%N)

    for (( teacher_id=0; teacher_id<nb_teachers; teacher_id++ )); do
        local t_start
        t_start=$(date +%s%N)
        echo "  [teacher $teacher_id / $((nb_teachers-1))]"
        local t_status="OK"
        python train_teachers.py \
            --dataset="$dataset" \
            --nb_teachers="$nb_teachers" \
            --teacher_id="$teacher_id" \
            --epochs="$teacher_epochs" \
            --batch_size="$BATCH_SIZE" \
            --lr="$LEARNING_RATE" \
            --data_dir="$DATA_DIR" \
            --train_dir="$train_dir" \
            2>&1 | tee -a "$log_file" || t_status="FAILED"
        local t_end
        t_end=$(date +%s%N)
        local t_wall_s
        t_wall_s=$(python3 -c "print(f'{($t_end - $t_start) / 1e9:.2f}')")
        echo "  → Teacher $teacher_id done in ${t_wall_s}s  [${t_status}]"
        printf "teacher\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$dataset" "$nb_teachers" "$teacher_epochs" "-" "-" \
            "$t_wall_s" "$t_status" "$(basename "$log_file"):teacher_${teacher_id}" \
            >> "$SUMMARY"
    done

    local end
    end=$(date +%s%N)
    local wall_s
    wall_s=$(python3 -c "print(f'{($end - $start) / 1e9:.2f}')")
    echo "  → All $nb_teachers teachers done in ${wall_s}s total"
    echo ""
}

# ---------------------------------------------------------------------------
# Run student training for one noise_scale value.
# ---------------------------------------------------------------------------
run_student() {
    local dataset="$1"
    local nb_teachers="$2"
    local teacher_epochs="$3"
    local noise_scale="$4"
    local eps_target="$5"
    local train_dir="$6"

    local tag="${dataset}_t${nb_teachers}_e${teacher_epochs}_eps${eps_target}_noise${noise_scale}"
    local log_file="$LOG_DIR/${tag}.log"
    local metrics_file="$LOG_DIR/${tag}_metrics.json"

    echo "--------------------------------------------------------------------"
    echo "  Student  |  dataset=$dataset  nb_teachers=$nb_teachers  teacher_epochs=$teacher_epochs"
    echo "  noise_scale=$noise_scale  (targeting ε ≈ $eps_target)"
    echo "  Log: $log_file"
    echo "--------------------------------------------------------------------"

    local start
    start=$(date +%s%N)

    local status="OK"
    python train_student.py \
        --dataset="$dataset" \
        --nb_teachers="$nb_teachers" \
        --stdnt_share="$STUDENT_QUERIES" \
        --noise_scale="$noise_scale" \
        --delta="$DELTA" \
        --epochs="$STUDENT_EPOCHS" \
        --batch_size="$BATCH_SIZE" \
        --lr="$LEARNING_RATE" \
        --data_dir="$DATA_DIR" \
        --train_dir="$train_dir" \
        --teachers_dir="$train_dir" \
        --metrics_file="$metrics_file" \
        --save_labels \
        2>&1 | tee "$log_file" || status="FAILED"

    local end
    end=$(date +%s%N)
    local wall_s
    wall_s=$(python3 -c "print(f'{($end - $start) / 1e9:.2f}')")

    echo ""
    echo "  → Finished in ${wall_s}s  [${status}]"
    echo ""

    printf "student\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$dataset" "$nb_teachers" "$teacher_epochs" "$noise_scale" "$eps_target" \
        "$wall_s" "$status" "$(basename "$log_file")" \
        >> "$SUMMARY"
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
TOTAL_START=$(date +%s%N)
total=0

for dataset in "${DATASETS[@]}"; do
    for nb_t in "${NB_TEACHERS[@]}"; do
        for teacher_epochs in "${TEACHER_EPOCHS_LIST[@]}"; do
            TRAIN_DIR="${BASE_TRAIN_DIR}/${dataset}_t${nb_t}_e${teacher_epochs}"
            mkdir -p "$TRAIN_DIR"

            # Train each teacher exactly once for this combo
            train_teachers "$dataset" "$nb_t" "$teacher_epochs" "$TRAIN_DIR"

            # Sweep noise scales for the student
            read -r -a scales <<< "${NOISE_SCALES[$nb_t]}"
            for i in "${!EPSILON_TARGETS[@]}"; do
                eps="${EPSILON_TARGETS[$i]}"
                noise="${scales[$i]}"
                run_student "$dataset" "$nb_t" "$teacher_epochs" "$noise" "$eps" "$TRAIN_DIR"
                total=$((total + 1))
            done
        done
    done
done

TOTAL_END=$(date +%s%N)
TOTAL_S=$(python3 -c "print(f'{($TOTAL_END - $TOTAL_START) / 1e9:.2f}')")

echo "===================================================================="
echo "  All experiments complete"
echo "  Student runs : $total"
echo "  Total time   : ${TOTAL_S}s"
echo "  Summary      : $SUMMARY"
echo "===================================================================="
