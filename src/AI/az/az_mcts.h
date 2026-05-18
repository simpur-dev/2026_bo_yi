#pragma once
#include "../board.h"
#include "az_node.h"
#include "az_types.h"
#include <vector>
#include <array>

// ========== PUCT 蒙特卡洛树搜索 ==========
//
// 一次迭代流程:
//
//   1. Selection
//      从根节点出发，根据 PUCT 分数一路选择子节点，直到到达叶节点或终局节点
//
//   2. Evaluation
//      到达叶节点后:
//        - 游戏已结束 → 根据比分计算精确价值
//        - 进入终局（只剩长链/环）→ 调用精确终局求解器
//        - 其他情况 → 调用策略价值网络（或启发式评估器）
//
//   3. Expansion
//      根据 policy 为所有合法动作创建子节点
//      每个子节点执行该动作 → 自动吃 C 型格 → 确定下一个玩家
//
//   4. Backup
//      把 value 沿路径回传，更新每个节点的 visits 和 valueSum
//      回传时根据节点 player 是否与叶节点 player 相同决定加减号

class AZMCTS
{
  public:
    // 执行 PUCT 搜索，返回根节点（调用方负责释放）
    // 使用 MCTSConfig 控制搜索行为（噪声、时间限制等）
    AZNode *search(const Board &board, int player, const MCTSConfig &config);

    // 兼容旧接口（默认 evaluation 模式，无噪声）
    AZNode *search(const Board &board, int player, int numSimulations, int timeLimitMs);

    // 选择最佳动作
    // temperature = 0: 贪心选访问次数最多的
    // temperature > 0: 按 visits^(1/T) 的概率采样（用于自对弈训练）
    int selectAction(const AZNode *root, float temperature = 0.0f) const;

    // 获取访问次数分布 π（用于生成训练数据）
    std::array<float, AZ_ACTION_SIZE> getVisitDistribution(const AZNode *root) const;

    // 收集搜索统计信息
    AZSearchStats getSearchStats(const AZNode *root, int elapsedMs) const;

  private:
    MCTSConfig config_; // 当前搜索配置

    // 单次 PUCT 迭代
    void simulate(AZNode *root);

    // 扩展叶节点: 根据 policy 创建所有合法动作的子节点
    void expand(AZNode *node, const NetworkOutput &output);

    // 回传价值: 沿路径更新 visits 和 valueSum
    void backup(const std::vector<AZNode *> &path, float value, int leafPlayer);

    // 检查并标记终局节点（游戏结束 / 只剩长链环）
    // 返回 true 表示该节点是终局，value 已写入 terminalValue
    bool checkAndMarkTerminal(AZNode *node) const;

    // 用精确终局求解器评估（仅在终局时调用）
    float endgameEvaluate(const Board &board, int player) const;
};
