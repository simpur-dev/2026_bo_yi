#include "define.h"
#include <functional>
#include <vector>
using std::vector;
#pragma once
class Board
{
  public:
    int map[11][11]{}; /**<棋盘*/
    int blackBox = 0;  /**<黑放所占格子*/
    int whiteBox = 0;  /**<白方所占格子*/
    Board();
    Board(int Array[LEN][LEN]); //构造函数

    void setBoard(int Array[LEN][LEN]);                   //*设置局面
    int move(int player, LOC l);                          /**<下棋*/
    void unmove(LOC l);                                   /**<模拟时悔棋*/
    LOC eatCBox(int player);                              /**<吃C型格*/
    void eatAllCTypeBoxes(int Player, vector<LOC> &pace); //*吃掉所有的C型格
    void eatAllCTypeBoxes(int Player);                    //*吃掉所有的C型格

    //局面估值
    bool getFreeBoxBool(int bx, int xy);      //*判断某个格子是否为自由格
    int getPlayerBoxes(int player) const;           //*得到某个玩家占领的格子的总数
    int getFreedom(int x, int y);             /**<得到自由度*/
    int getFreedom(LOC l);                    /**<得到自由度*/
    int getFreeEdgeNum();                     //*得到自由边的数量
    bool getCTypeBoxBool(int bx, int by);     //*判断局面中是否有C型格
    bool getLongCTypeBoxBool(int bx, int by); //*判断一个格子是否是一条长死格的起点
    bool getLongCTypeBoxExist();              //*判断局面中是否存在长C型链
    bool ifEarnBox(LOC l);                    /**<是否赢得格子*/
    bool ifHaveSafeEdge();
    bool isFreeLine(LOC l) const;        /**<是否是自由边*/
    bool isFreeLine(int i, int j) const; /**<是否是自由边*/
    bool ifMakeCBox(LOC l);              /**<是否制造了C型格*/
    int getWinner() const;               /**<返回赢家*/
    bool ifEnd() const;                  /**<判断游戏是否结束*/

    LOC getPublicSide(LOC a, LOC b); //获得两个格子的公共边，传入的坐标为(5*5)

    //计算
    LOC getDoubleCrossLoc(int Player); //*查找一个doublecross的制作方法，返回值为该边的坐标
    bool getCTypeBoxLimit(int Player, vector<LOC> &pace); //*限制版吃C型格
    int getFilterMoveNum();                               //*得到所有的过滤可行边的数量
};

bool isOdd(int num);  //判断是否为奇数
bool isEven(int num); //判断是否为偶数
void boardCopy(int Source[LEN][LEN], int target[LEN][LEN]);