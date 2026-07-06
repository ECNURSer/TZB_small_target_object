#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="${CONDA_ENV:-yolo26-obb}"
export YOLO_CONFIG_DIR="$PROJECT/.ultralytics"
export MPLBACKEND="${MPLBACKEND:-Agg}"
mkdir -p "$YOLO_CONFIG_DIR"
MODE="${1:-help}"
shift || true

run_python() {
    conda run --no-capture-output -n "$ENV_NAME" python "$@"
}

case "$MODE" in
    train-n|train-s|train-m)
        SIZE="${MODE#train-}"
        run_python "$PROJECT/train.py" --model "$SIZE" "$@"
        ;;
    test)
        run_python "$PROJECT/evaluate_test.py" "$@"
        ;;
    competition)
        run_python "$PROJECT/evaluate_competition.py" "$@"
        ;;
    predict)
        run_python "$PROJECT/predict.py" "$@"
        ;;
    tensorboard)
        LOGDIR=""
        PORT="6006"
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --logdir)
                    [[ $# -ge 2 ]] || { echo "错误: --logdir 缺少目录" >&2; exit 2; }
                    LOGDIR="$2"
                    shift 2
                    ;;
                --port)
                    [[ $# -ge 2 ]] || { echo "错误: --port 缺少端口号" >&2; exit 2; }
                    PORT="$2"
                    shift 2
                    ;;
                *)
                    echo "错误: tensorboard 不支持参数 $1" >&2
                    exit 2
                    ;;
            esac
        done
        [[ -n "$LOGDIR" ]] || {
            echo "错误: 必须明确指定 --logdir" >&2
            echo "示例: bash run.sh tensorboard --logdir $PROJECT/runs/yolo26n_obb_fold0" >&2
            exit 2
        }
        [[ "$LOGDIR" = /* ]] || LOGDIR="$PROJECT/$LOGDIR"
        [[ -d "$LOGDIR" ]] || { echo "错误: TensorBoard 日志目录不存在: $LOGDIR" >&2; exit 2; }
        EVENT_FILE="$(find "$LOGDIR" -type f -name 'events.out.tfevents.*' -print -quit)"
        [[ -n "$EVENT_FILE" ]] || {
            echo "错误: 目录中没有 events.out.tfevents.* 文件: $LOGDIR" >&2
            echo "请先启动训练，等日志文件生成后再启动 TensorBoard。" >&2
            exit 2
        }
        echo "TensorBoard 日志目录: $LOGDIR"
        echo "检测到 events 文件: $EVENT_FILE"
        echo "访问地址: http://服务器IP:$PORT"
        conda run --no-capture-output -n "$ENV_NAME" tensorboard --logdir "$LOGDIR" --port "$PORT" --bind_all
        ;;
    summary)
        run_python "$PROJECT/tools/summarize_results.py"
        ;;
    help|*)
        cat <<'EOF'
用法:
  bash run.sh train-n --fold 0
  bash run.sh train-s --fold 0
  bash run.sh train-m --fold 0
  bash run.sh tensorboard --logdir runs/yolo26n_obb_fold0_balanced_focal_700ep_b64 --port 6007
  bash run.sh test --model m --fold 0 --weights runs/yolo26m_obb_fold0_balanced_focal/weights/best.pt --imgsz 1024 --device 7
  bash run.sh competition --weights runs/yolo26m_obb_fold0_balanced_focal/weights/best.pt --device 7 --imgsz 1280 --max-det 600 --cache runs/competition/m.json --output runs/competition/m_metrics.json
  bash run.sh predict --weights runs/yolo26m_obb_fold0_balanced_focal/weights/best.pt --source image.tif --device 7
  bash run.sh summary
EOF
        ;;
esac
