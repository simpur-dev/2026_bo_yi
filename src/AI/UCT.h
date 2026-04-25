#include "Node.h"
#include "board.h"
#include <mutex>
#include <random>
#include <thread>

#ifndef DOTS_AND_BOXES_UCT_H
#define DOTS_AND_BOXES_UCT_H

int getBoardWinner(Board &CB, int LatterPlayer);                          //*得到或者预测当前局面胜利者
int getBoardWinner(Board &CB, int LatterPlayer, int FaOwner, int &score); //*得到或者预测当前局面胜利者

// FilterMC，
float getFilterMCEvalution(Board &CB, int NextPlayer, int Winner);
//*在扩展子节点时，为子节点初次评估一个收益
int getFilterMCWinner(Board &CB, int NextPlayer);             //*返回模拟的胜负结果
int getFilterMCWinner(Board &CB, int NextPlayer, int &score); //*返回模拟的胜负结果
int rndFilterTurn(Board &CB, int Player);                     //*用于模拟时随机占边

float UCTProcess(Node &B, int &Total);                  //*uct搜索
void UCTMove(Board &CB, int Player, vector<LOC> &pace); //*用UCT算法进行移动

void deleteUCTNode(Node *Root); //*用于释放空间
void deleteUCTTree(Node &Root);
void UCTMoveWithSacrifice(Board &CB, int Player, vector<LOC> &pace); //*包括UCT搜索前的预处理
void latterSituationMove(Board &CB, int Player, vector<LOC> &pace); //*基于特殊结构体的决策，一般用于后期
void gameTurnMove(Board &CB, int Player, int *status, vector<LOC> &pace); //*根据前后期自动移动

#endif // DOTS_AND_BOXES_UCT_H