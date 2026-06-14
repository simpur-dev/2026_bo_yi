#!/usr/bin/env bash
# =============================================================================
# 推送/拉取 RL 训练数据 (本机 ↔ 服务器)
# =============================================================================
#
# 用法:
#   # 从本机推 baseline 备份到服务器
#   bash tools/rsync_data.sh push <user>@<server>:<path>
#
#   # 从服务器拉 RL 训练结果回本机
#   bash tools/rsync_data.sh pull <user>@<server>:<path>
#
# 默认路径:
#   本机: data/backup_pre_rl_2026-06-14/  →  服务器: ~/2026_bo_yi/data/backup_pre_rl_2026-06-14/
#   服务器: ~/2026_bo_yi/data/rl_iter1/   →  本机: data/rl_iter1/
#
# 提示:
#   - 不进入 git 仓库, 避免 173MB 备份被 commit
#   - 用 --exclude "*.pyc" / "__pycache__" 排除编译产物
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

DIRECTION="${1:-}"
REMOTE="${2:-user@server:~/2026_bo_yi}"

if [[ -z "${DIRECTION}" ]]; then
    echo "用法:"
    echo "  $0 push <user>@<server>:<path>   推本机备份到服务器"
    echo "  $0 pull <user>@<server>:<path>   拉服务器 RL 输出回本机"
    echo ""
    echo "默认服务器路径: ${REMOTE}"
    echo ""
    echo "示例 (push 备份):"
    echo "  $0 push user@10.0.0.5:~/2026_bo_yi"
    echo ""
    echo "示例 (pull RL 输出):"
    echo "  $0 pull user@10.0.0.5:~/2026_bo_yi"
    exit 1
fi

EXCLUDES=(
    "--exclude=__pycache__"
    "--exclude=*.pyc"
    "--exclude=*.pt.bak"
    "--exclude=.DS_Store"
    "--exclude=Thumbs.db"
)

case "${DIRECTION}" in
    push)
        echo "===== 推 backup_pre_rl_2026-06-14/ → ${REMOTE} ====="
        rsync -avz --progress "${EXCLUDES[@]}" \
            "data/backup_pre_rl_2026-06-14/" \
            "${REMOTE}/data/backup_pre_rl_2026-06-14/"
        echo ""
        echo "===== 推 train/ + eval/ + tools/run_rl_on_server.sh → ${REMOTE} (这是 git 管的, 但推送一次也行) ====="
        echo "  提示: 训练代码请走 git push, 不要 rsync"
        echo "    git add train/rl_finetune_ppo.py tools/run_rl_on_server.sh eval/eval_vs_baseline.py"
        echo "    git commit -m 'feat: add RL fine-tune (PPO) + server deploy scripts'"
        echo "    git push origin main"
        ;;
    pull)
        echo "===== 拉 ${REMOTE}/data/rl_iter*/ → data/rl_iter*/ ====="
        rsync -avz --progress "${EXCLUDES[@]}" \
            "${REMOTE}/data/" \
            "data/"
        echo ""
        echo "  拉回的文件:"
        echo "    data/rl_iter*/rl_log.jsonl         训练日志"
        echo "    data/rl_iter*/checkpoints/*.pt     模型快照"
        echo "    data/rl_iter*/evals/*.json         vs baseline 评估结果"
        echo "    data/rl_iter*/rollouts/*.jsonl     selfplay 数据"
        echo ""
        echo "  下一步: 在本机跑评估 vs 基线"
        echo "    python eval/eval_vs_baseline.py \\"
        echo "      --candidate data/rl_iter1/checkpoints/candidate_iter_0008.pt \\"
        echo "      --baseline  data/backup_pre_rl_2026-06-14/backward_best_model_white.onnx \\"
        echo "      --games 20 --sims 400"
        ;;
    *)
        echo "ERROR: direction 必须是 'push' 或 'pull', 得到: ${DIRECTION}"
        exit 1
        ;;
esac
