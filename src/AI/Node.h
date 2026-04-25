#include "board.h"
#include "define.h"
#ifndef DOTS_AND_BOXES_NODE_H
#define DOTS_AND_BOXES_NODE_H

class Node : public Board
{
  public:
    // UCT使用的值
    int Owner;            //所属玩家
    int Times;            //被尝试过的次数
    int BoardWinner;      //这个局面的胜利者
    int ExistChild;       //子节点当前的数目
    int TotalChild;       //子节点的总数。
    float AvgValue;       //对父节点而言，这个节点的平均收益
    LOC NodeMoves[60];    //这个子节点所有可能的招式
    Node *ChildNodes[60]; //指向第一个子节点的指针

    Node(); //构造函数
    Node(Node &other);
    Node(int Player, int Array[LEN][LEN], bool GetCBox); //*构造函数

    //功能函数
    float refreshAvgValue();      //*更新平均收益
    float getUCBValue(int Total); //*根据现在的平均收益而获得UCB值
    Node *expandUCTNode();        //*扩展节点
};

#endif // DOTS_AND_BOXES_NODE_H
