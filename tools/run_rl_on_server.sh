#!/usr/bin/env bash
# =============================================================================
# 在服务器上跑 RL Fine-tune (PPO) 的部署脚本
# =============================================================================
#
# 使用场景:
#   1. 你已经把代码 git push 到 GitHub (simpur-dev/2026_bo_yi)
#   2. 你已经把 data/backup_pre_rl_2026-06-14/ 用 rsync 传到了服务器
#   3. 服务器上已经有 PyTorch + CUDA
#
# 在服务器上的执行流程:
#
#   # 1. 拉最新代码
#   cd ~/2026_bo_yi
#   git pull origin main
#
#   # 2. 编译 selfplay + evaluate_ai (用 Unix Makefiles)
#   cmake -B build -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DUSE_ONNX=ON
#   cmake --build build --target backward_selfplay evaluate_ai -j$(nproc)
#
#   # 3. 跑 RL fine-tune (示例 20 轮, 每轮 50 局)
#   bash tools/run_rl_on_server.sh \
#     --data-out data/rl_iter1 \
#     --total-iterations 20 \
#     --games-per-iter 50 \
#     --sims 400 \
#     --eval-interval 2
#
#   # 4. 训练中查看进度
#   tail -f data/rl_iter1/rl_log.jsonl
#   tail -f data/rl_iter1/evals/*.json
#
#   # 5. 同步回本机
#   rsync -avz --progress user@server:~/2026_bo_yi/data/rl_iter1/ data/rl_iter1/
#
# 红线:
#   - 本脚本只读不写 3rdparty/onnxruntime_cpu/ (只走 LD_LIBRARY_PATH)
#   - 本脚本不读不写 simpur/ ruikang/
#   - baseline 路径必须位于 data/backup_pre_rl_2026-06-14/ 之下
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# 默认参数
DATA_OUT="data/rl_iter1"
TOTAL_ITERS=20
GAMES_PER_ITER=50
SIMS=400
EVAL_INTERVAL=2
EVAL_GAMES=20
ARCH="resnet_s"
DEVICE="cuda"
PPO_EPOCHS=4
PPO_CLIP=0.2
LR=3e-5
SEED=42
WALLTIME_H=0
RESUME=""

# baseline 默认路径 (相对 PROJECT_ROOT)
BASELINE_BLACK="data/backup_pre_rl_2026-06-14/backward_best_model_black.pt"
BASELINE_WHITE="data/backup_pre_rl_2026-06-14/backward_best_model_white.pt"

usage() {
    grep '^#' "$0" | head -n 30
    echo ""
    echo "Options (覆盖默认值):"
    echo "  --data-out DIR             RL 输出目录 (默认: ${DATA_OUT})"
    echo "  --total-iterations N       总迭代数 (默认: ${TOTAL_ITERS})"
    echo "  --games-per-iter N         每轮 selfplay 局数 (默认: ${GAMES_PER_ITER})"
    echo "  --sims N                   MCTS 模拟数 (默认: ${SIMS})"
    echo "  --eval-interval N          每 N 轮做评估 (默认: ${EVAL_INTERVAL})"
    echo "  --eval-games N             评估对局数 (默认: ${EVAL_GAMES})"
    echo "  --arch NAME                mlp/resnet_s/resnet_m/resnet_l (默认: ${ARCH})"
    echo "  --device NAME              cuda/cpu (默认: ${DEVICE})"
    echo "  --ppo-epochs N             PPO 内循环 epoch (默认: ${PPO_EPOCHS})"
    echo "  --ppo-clip F               PPO clip epsilon (默认: ${PPO_CLIP})"
    echo "  --lr F                     学习率 (默认: ${LR})"
    echo "  --seed N                   随机种子 (默认: ${SEED})"
    echo "  --max-walltime-h N         墙钟时间上限 (小时, 0=不限)"
    echo "  --resume CKPT              恢复训练的 checkpoint"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-out) DATA_OUT="$2"; shift 2 ;;
        --total-iterations) TOTAL_ITERS="$2"; shift 2 ;;
        --games-per-iter) GAMES_PER_ITER="$2"; shift 2 ;;
        --sims) SIMS="$2"; shift 2 ;;
        --eval-interval) EVAL_INTERVAL="$2"; shift 2 ;;
        --eval-games) EVAL_GAMES="$2"; shift 2 ;;
        --arch) ARCH="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --ppo-epochs) PPO_EPOCHS="$2"; shift 2 ;;
        --ppo-clip) PPO_CLIP="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --max-walltime-h) WALLTIME_H="$2"; shift 2 ;;
        --resume) RESUME="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# ---------- 0. 环境检查 ----------
echo "===== [0/4] 环境检查 ====="
echo "  Python: $(python3 --version 2>&1)"
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>&1)"
if [[ "${DEVICE}" == "cuda" ]]; then
    echo "  CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())' 2>&1)"
    if python3 -c 'import torch; exit(0 if torch.cuda.is_available() else 1)'; then
        echo "  GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>&1)"
    else
        echo "  WARNING: CUDA 不可用, 建议加 --device cpu 或检查环境"
    fi
fi

# baseline 路径
if [[ ! -f "${BASELINE_BLACK}" ]]; then
    echo "ERROR: baseline black 不存在: ${BASELINE_BLACK}"
    echo "  请先 rsync 备份目录到服务器:"
    echo "    rsync -avz --progress <本机>:/d/Myfiles/Computer_chess/2026_chess/2026_bo_yi/data/backup_pre_rl_2026-06-14/ <server>:~/2026_bo_yi/data/backup_pre_rl_2026-06-14/"
    exit 2
fi
if [[ ! -f "${BASELINE_WHITE}" ]]; then
    echo "ERROR: baseline white 不存在: ${BASELINE_WHITE}"
    exit 2
fi
echo "  baseline-black: ${BASELINE_BLACK}"
echo "  baseline-white: ${BASELINE_WHITE}"

# ---------- 1. 编译 selfplay + evaluate_ai ----------
echo ""
echo "===== [1/4] 编译 C++ 工具 ====="
if [[ ! -d "build" ]]; then
    echo "  cmake 配置中..."
    cmake -B build -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DUSE_ONNX=ON
fi
echo "  构建 backward_selfplay + evaluate_ai ..."
cmake --build build --target backward_selfplay evaluate_ai -j"$(nproc)"

# ---------- 2. 准备输出目录 ----------
echo ""
echo "===== [2/4] 准备输出目录 ====="
mkdir -p "${DATA_OUT}/rollouts"
mkdir -p "${DATA_OUT}/checkpoints"
mkdir -p "${DATA_OUT}/evals"
echo "  ${DATA_OUT}/"

# ---------- 3. 跑 RL ----------
echo ""
echo "===== [3/4] RL Fine-tune (PPO) ====="
EXTRA_ARGS=""
if [[ -n "${RESUME}" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --resume ${RESUME}"
fi
if [[ "${WALLTIME_H}" != "0" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --max-iters-walltime-h ${WALLTIME_H}"
fi

python3 train/rl_finetune_ppo.py \
    --baseline-black "${BASELINE_BLACK}" \
    --baseline-white "${BASELINE_WHITE}" \
    --data-out "${DATA_OUT}" \
    --selfplay-exe ./build/backward_selfplay \
    --eval-exe ./build/evaluate_ai \
    --arch "${ARCH}" \
    --device "${DEVICE}" \
    --games-per-iter "${GAMES_PER_ITER}" \
    --sims "${SIMS}" \
    --start-step 0 \
    --total-iterations "${TOTAL_ITERS}" \
    --eval-interval "${EVAL_INTERVAL}" \
    --eval-games "${EVAL_GAMES}" \
    --ppo-epochs "${PPO_EPOCHS}" \
    --ppo-clip "${PPO_CLIP}" \
    --lr "${LR}" \
    --seed "${SEED}" \
    --rollback-winrate 0.40 \
    --rollback-patience 3 \
    ${EXTRA_ARGS}

# ---------- 4. 收尾 ----------
echo ""
echo "===== [4/4] 完成 ====="
echo "  RL 输出: ${DATA_OUT}/"
echo "  日志:    ${DATA_OUT}/rl_log.jsonl"
echo "  评估:    ${DATA_OUT}/evals/"
echo "  最佳 checkpoint: ${DATA_OUT}/checkpoints/candidate_iter_*.pt"
echo ""
echo "  同步回本机:"
echo "    rsync -avz --progress user@server:~/2026_bo_yi/data/${DATA_OUT}/ data/${DATA_OUT}/"
