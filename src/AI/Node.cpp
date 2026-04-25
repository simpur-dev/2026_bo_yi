#include "Node.h"
#include "UCT.h"
#include "assess.h"
#include <cmath>

Node::Node()
{
    // 初始化棋盘
    Owner = Times = BoardWinner = ExistChild = TotalChild = 0;
    AvgValue = 0;
}

Node::Node(Node &other) : Board(other)
{
    Owner = other.Owner;
    Times = other.Times;
    BoardWinner = other.BoardWinner;
    ExistChild = other.ExistChild;
    TotalChild = other.TotalChild;
    AvgValue = other.AvgValue;
    for (int i = 0; i < 60; ++i)
    {
        NodeMoves[i] = other.NodeMoves[i];
        ChildNodes[i] = other.ChildNodes[i];
    }
}

Node::Node(int Player, int Array[LEN][LEN], bool GetCBox) : Board(Array)
{

    if (GetCBox)
        eatAllCTypeBoxes(Player); // 先全部吃掉C型格，此时的Node自动跃迁为Gotten Node

    // 然后开始初始化数据
    Owner = Player;                             // 这个节点的拥有者是传入的数据
    Times = 1;                                  // 创建时候这个节点已经被访问一次了
    Board Win(map);                             // 创建当前局面的判断点
    BoardWinner = getBoardWinner(Win, -Player); // 这个局面的胜利者，注意此时的先手是我方
    ExistChild = 0;                             // 已经存在的子节点数为0

    int F = Win.getFreeEdgeNum();
    if (F >= UCT_FILTER_RANGE)
    {
        BoxBoard Test(map);
        TotalChild = Test.getFreeMoves(NodeMoves); // 提前计算总节点数
    }
    else
    {
        BoxBoard Test(map);
        TotalChild = Test.getFilterMoves(NodeMoves); // 提前计算总节点数
    }
}

float Node::refreshAvgValue()
{
    float Value = 0;
    // 首先计算平均收益
    for (int i = 0; i < ExistChild; i++)
    {
        Value += ChildNodes[i]->AvgValue * ChildNodes[i]->Times / Times;
    }
    AvgValue = 1 - Value; // 当前节点的平均收益必定是(1 - 子节点的平均收益 )
    return AvgValue;
}

float Node::getUCBValue(int Total)
{
    return AvgValue + sqrt((log(Total) * 2) / Times);
}

Node *Node::expandUCTNode()
{
    // 创建新节点
    Board CB(map);
    CB.move(Owner, {NodeMoves[ExistChild].first, NodeMoves[ExistChild].second});
    Node *NewB = new Node(-Owner, CB.map, true);

    // 做一次MC评估
    CB.setBoard(NewB->map);
    NewB->AvgValue = (getFilterMCEvalution(CB, -Owner, Owner) + 13) / 26;
    return NewB; // 返回NewB的地址
}
