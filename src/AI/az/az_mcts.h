#pragma once
#include "../board.h"
#include "az_node.h"
#include "az_types.h"
#include <vector>
#include <array>

// PUCT 蒙特卡洛树搜索
class AZMCTS
{
  public:
    // 对当前局面执行 PUCT 搜索，返回根节点
    // board: 已吃完 C 型格的局面
    // player: 当前玩家
    // numSimulations: 模拟次数
    // timeLimitMs: 时间上限（毫秒），0 表示不限时
    AZNode *search(const Board &board, int player, int numSimulations, int timeLimitMs);

    // 从搜索结果中选择最佳动作
    // temperature: 0.0 贪心选最多访问次数, > 0 按访问次数的概率采样
    int selectAction(const AZNode *root, float temperature = 0.0f) const;

    // 获取根节点子节点的访问次数分布（用于训练数据）
    std::array<float, AZ_ACTION_SIZE> getVisitDistribution(const AZNode *root) const;

  private:
    // 单次 PUCT 迭代：选择 -> 评估/扩展 -> 回传
    void simulate(AZNode *root);

    // 扩展叶节点
    void expand(AZNode *node, const NetworkOutput &output);

    // 回传价值
    void backup(const std::vector<AZNode *> &path, float value, int leafPlayer);

    // 判断局面是否为终局（只剩长链/环，无 Filter 可行边）
    bool isEndgame(const Board &board, int player) const;

    // 用精确终局求解器评估
    float endgameEvaluate(const Board &board, int player) const;
};
