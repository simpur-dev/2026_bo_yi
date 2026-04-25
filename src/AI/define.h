/*
 *此头文件是存放一些常用的宏定义
 */
#include <utility>

#ifndef DOTS_AND_BOXES_DEFINE_H
#define DOTS_AND_BOXES_DEFINE_H

constexpr int Dir[4][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};

// 各种链的编号  未定义，长链，短链，环，预备环，死链，死环
enum ChainType
{
    NotDefine,
    LongChain,
    ShortChain,
    Circle,
    PreCircle,
    DeadChain,
    DeadCircle
};

// 点
constexpr int DOT = 10;

// 线，HENG是横线，SHU是竖线
//  OCCLINE代表occupied line，代表被占据的线
constexpr int HENG = 20;
constexpr int SHU = 30;
constexpr int OCCLINE = 40;

// 格子，大于isBOX代表是格子
constexpr int BOX = 0;

// 为了和格子对应
constexpr int EMPTY = 0;
constexpr int BLACK = 1;
constexpr int WHITE = -1;

constexpr double C = 1; // UCB 常数
// 定义格子类型
constexpr int FULLBOX = 0;  // 满格
constexpr int DEADBOX = 1;  // 死格
constexpr int CHAINBOX = 2; // 链格
constexpr int FREEBOX = 3;  // 自由格
constexpr int LEN = 11;     // 棋盘数组长度
constexpr int BOXLEN = 5;   // 格子数组长度
constexpr int BOXNUM = 25;

// UCT
constexpr int UCT_MC_TIMES = 1;
constexpr int UCT_TIMES = 15000000;
constexpr int UCT_LIMIT_TIME = 1;
constexpr int UCT_FILTER_RANGE = 30;

// 一些常用的
// 不使用using namespace std,是为了防止和std空间里的命名冲突
using LOC = std::pair<int, int>;

// step用于记录棋谱
struct Step
{
    int player = EMPTY;
    LOC action;
};

#endif // DOTS_AND_BOXES_DEFINE_H