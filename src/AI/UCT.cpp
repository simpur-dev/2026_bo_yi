#include "UCT.h"
#include "Node.h"
#include "assess.h"
#include "board.h"
#include "define.h"
#include <ctime>
#include <iostream>
#include <random>

std::mt19937_64 rng(std::random_device{}());

int getBoardWinner(Board &CB, int LatterPlayer)
{
    BoxBoard Advanced(CB);
    if (Advanced.getFilterMoveNum() != 0)
        return EMPTY;
    else
        return Advanced.getBoardWinner(LatterPlayer);
}

int getBoardWinner(Board &CB, int LatterPlayer, int FaOwner, int &score)
{
    BoxBoard Advanced(CB);
    if (Advanced.getFilterMoveNum() != 0) //未到终局
    {
        score = Advanced.getPlayerBoxes(FaOwner); // score为该节点的父结点拥有者即FaOwner得到格子数
        return 0;
    }
    //到终局
    int w = Advanced.getBoardWinner(LatterPlayer, FaOwner, score); // score为该节点的父结点拥有者即FaOwner得到格子数
     return w;
}

int getFilterMCWinner(Board &CB, int NextPlayer)
{
    int player = NextPlayer;
    while (CB.getFreeEdgeNum() != 0) //当还存在自由边的时候
    {
        player = rndFilterTurn(CB, player); //#传入后续玩家#
    }
    int W = getBoardWinner(CB, -player);
    return W;
}

int getFilterMCWinner(Board &CB, int NextPlayer, int &score) //第二个参数为当前节点拥有者
{
    int player = NextPlayer;
    while (CB.getFreeEdgeNum() != 0) //当还存在自由边的时候
    {
        player = rndFilterTurn(CB, player); //#传入后续玩家#
    }
    int W = getBoardWinner(CB, -player, -NextPlayer, score); //第三个参数为当前节点的父结点拥有者
    return W;
}

float getFilterMCEvalution(Board &CB, int NextPlayer, int Winner)
{
    Board MCB = CB; //先复制一个棋盘
    int MCE = 0;    //每次模拟所得收益值的总和
    // int threadnum = 11;
    // thread ths[11];
    //当前版本为11线程并行模拟
    for (int i = 0; i < UCT_MC_TIMES; i++)
    {
        int CNT = 0; //每次模拟该节点的父结点拥有者获得格子数
        int WIN = 0; //每次模拟的结果，0或1
        if (getFilterMCWinner(MCB, NextPlayer, CNT) == Winner) //传入的是当前节点拥有者
            WIN++;
        MCE += WIN + CNT - 13;
    }
    // for (int i = 0; i < threadnum; i++)
    // {
    //     ths[i] = thread(multi_thread_func, ref(MCB), ref(MCE), NextPlayer, Winner, UCT_FILTER_RANGE);
    // }
    // for (int i = 0; i < threadnum; i++)
    // {
    //     ths[i].join();
    // }
    // float score = ((float)MCE) / ((float)threadnum);
    float score = ((float)MCE) / ((float)UCT_MC_TIMES);
    return score;
}

float UCTProcess(Node &B, int &Total) //#Total 尝试次数#
{
    B.Times++;              //访问的次数增加一次
    if (B.BoardWinner != 0) //如果游戏已经结束了#叶节点#
    {
        if (B.BoardWinner == B.Owner)
            B.AvgValue = 0; //如果在这个节点游戏结束了，判定收益。
        else
            B.AvgValue = 1; //如果在这个节点游戏结束了，判定收益
        Total++;            //如果搜索到游戏结束的节点，则本次迭代结束。
        return B.AvgValue;
    }
    if (B.ExistChild < B.TotalChild) //如果还有未尝试过的节点
    {
        Total++;                                        //基准情形，本次迭代结束，尝试次数+1。
        B.ChildNodes[B.ExistChild] = B.expandUCTNode(); //扩展一个子节点
        B.ExistChild++;                                 //子节点的数目自增1
        B.refreshAvgValue();                            //刷新收益
        return B.AvgValue;
    }
    else //说明没有未尝试过的节点
    {
        int BestNodeNum = 0;
        double BestUCBValue = 0;
        double UCBValue[60];
        for (int i = 0; i < B.TotalChild; i++)
        {
            UCBValue[i] = B.ChildNodes[i]->getUCBValue(Total);
            if (UCBValue[i] >= BestUCBValue)
            {
                BestNodeNum = i;
                BestUCBValue = UCBValue[i];
            }
        }
        UCTProcess(*B.ChildNodes[BestNodeNum], Total);
        B.refreshAvgValue();
        return B.AvgValue;
    }
}

void UCTMove(Board &CB, int Player, vector<LOC> &pace)
{
    Node UCTB(Player, CB.map, true); //根据当前局面创建UCT的根节点
    if (UCTB.BoardWinner == 0)
    {
        int Total = 0;           // UCT的次数函数
        clock_t start = clock(); //设置计时器的变量

        for (int i = 0; i < UCT_TIMES; i++) //迭代一定次数
        {
            UCTProcess(UCTB, Total);

            if ((clock() - start) / CLOCKS_PER_SEC >= UCT_LIMIT_TIME)
                break;
        }
        //判定最佳收益
        int BestNodeNum = 0;
        float BestAvgValue = 0;
        //        int LargerTimesNodeNum = 0;
        //        int LargerTimesValue = 0;
        for (int i = 0; i < UCTB.ExistChild; i++)
        {
            if (UCTB.ChildNodes[i]->AvgValue >= BestAvgValue)
            {
                BestNodeNum = i;
                BestAvgValue = UCTB.ChildNodes[i]->AvgValue;
            }
            //            if (UCTB.ChildNodes[i]->Times >= LargerTimesValue)
            //            {
            //                LargerTimesNodeNum = i;
            //                LargerTimesValue = UCTB.ChildNodes[i]->Times;
            //            }
        }

        CB.move(Player, {UCTB.NodeMoves[BestNodeNum].first, UCTB.NodeMoves[BestNodeNum].second});
        pace.emplace_back(UCTB.NodeMoves[BestNodeNum].first, UCTB.NodeMoves[BestNodeNum].second); //**记录步伐
        deleteUCTTree(UCTB);
    }
    else
    {
        CB.eatAllCTypeBoxes(Player, pace);     //**记录步伐
        latterSituationMove(CB, Player, pace); //**记录步伐
    }
}

void deleteUCTNode(Node *Root)
{
    for (int i = 0; i < Root->ExistChild; i++)
    {
        deleteUCTNode(Root->ChildNodes[i]);
        delete Root->ChildNodes[i];
    }
}
void deleteUCTTree(Node &Root)
{
    for (int i = 0; i < Root.ExistChild; i++)
    {
        deleteUCTNode(Root.ChildNodes[i]);
        delete Root.ChildNodes[i];
    }
}

void UCTMoveWithSacrifice(Board &CB, int Player, vector<LOC> &pace)
{

    BoxBoard Dead(CB);
    bool DeadChain = Dead.getDeadChainExist();
    bool DeadCircle = Dead.getDeadCircleExist();
    if (DeadCircle || DeadChain) //有环的情况优先处理
    {
        int SacrificeBoxNum;
        if (DeadChain)
            SacrificeBoxNum = 2;
        if (DeadCircle)
            SacrificeBoxNum = 4;

        //假设全部都吃掉了
        Dead.eatAllCTypeBoxes(Player);              //此处传的第二个参数无意义
        LOC BoxNum = Dead.getEarlyRationalBoxNum(); // x是先手的，y是后手的
        //现在根据接下来局面的估值函数来进行分析
        if (BoxNum.first - BoxNum.second <= SacrificeBoxNum) //放弃控制
        {
            CB.eatAllCTypeBoxes(Player, pace); //**记录步伐
            UCTMove(CB, Player, pace);         //**记录步伐
        }
        else
        {
            //牺牲，此时必有死长链或死环
            LOC DCMove;
            if (SacrificeBoxNum == 2) //然后处理死链
            {
                //首先吃到贪婪的临界点
                for (;;)
                {
                    Board Test = CB;
                    Test.eatCBox(Player); //模拟走一步
                    BoxBoard Dead(Test);
                    if (Dead.getDeadChainExist())
                    {
                        LOC t = CB.eatCBox(Player);
                        pace.emplace_back(t); //*记录步伐
                    }
                    else
                        break;
                }
                //然后开始考虑DoubleCross
                DCMove = CB.getDoubleCrossLoc(Player);
            }
            else
            {
                //首先吃到贪婪的临界点
                for (;;)
                {
                    Board Test = CB;
                    Test.eatCBox(Player);
                    BoxBoard Dead(Test);
                    if (Dead.getDeadCircleExist())
                    {
                        LOC t = CB.eatCBox(Player);
                        pace.emplace_back(t); //**记录步伐
                    }
                    else
                        break;
                }
                //然后开始考虑DoubleCross
                DCMove = CB.getDoubleCrossLoc(Player);
            }

            CB.move(Player, {DCMove.first, DCMove.second});
            pace.emplace_back(DCMove); //**记录步伐
            for (;;)                   //吃掉所有死格
            {
                if (!CB.getCTypeBoxLimit(Player, pace)) //**记录步伐
                    break;
            }
            //牺牲结束
        }
    }
    else //正常UCT移动
    {
        CB.eatAllCTypeBoxes(Player, pace); //**记录步伐
        UCTMove(CB, Player, pace);         //**记录步伐
    }
}

void latterSituationMove(Board &CB, int Player, vector<LOC> &pace)
{
    //后期算法，此时只有长链和环。
    if (CB.getLongCTypeBoxExist())
    {
        //已有打开的长链，根据牺牲与不牺牲之后的理性状态决定是否牺牲。
        BoxBoard Dead(CB);
        int SacrificeBoxNum = 0;
        if (Dead.getDeadChainExist())
            SacrificeBoxNum = 2;
        if (Dead.getDeadCircleExist())
            SacrificeBoxNum = 4;

        //假设全部都吃掉了
        Dead.eatAllCTypeBoxes(Player);              //此处传的第二个参数无意义
        LOC BoxNum = Dead.getRationalStateBoxNum(); // x是先手的，y是后手的
        // cout << "BoxNum:" << BoxNum.first << " " << BoxNum.second << endl;
        // cout << "Sacri:" << SacrificeBoxNum << endl;
        //现在根据接下来局面能得到的格子数来进行分析
        //假设在当前链全被消灭后后手可以拿到x个，先手可以拿到y个。我方全吃后可以拿到y+n，对方拿到x个。若x-y<n则我方全吃
        if (BoxNum.first - BoxNum.second <= SacrificeBoxNum || Dead.getWinner() != EMPTY)
        {
            //放弃控制
            // cout << "Winner:" << Dead.getWinner() << endl;
            CB.eatAllCTypeBoxes(Player, pace); //**记录步伐
            if (Dead.getWinner() == EMPTY)
                latterSituationMove(CB, Player, pace); //**记录步伐
        }
        else
        {
            //牺牲，此时必有死长链或死环
            LOC DCMove;
            if (SacrificeBoxNum == 2) //然后处理死链
            {
                //首先吃到贪婪的临界点
                for (;;)
                {
                    Board Test = CB;
                    Test.eatCBox(Player); //模拟走一步
                    BoxBoard Dead(Test);
                    if (Dead.getDeadChainExist())
                    {
                        LOC tmp = CB.eatCBox(Player);
                        pace.emplace_back(tmp); //**记录步伐
                    }
                    else
                        break;
                }
                //然后开始考虑DoubleCross
                DCMove = CB.getDoubleCrossLoc(Player);
            }
            else
            {
                //首先吃到贪婪的临界点
                for (;;)
                {
                    Board Test = CB;
                    Test.eatCBox(Player);
                    BoxBoard Dead(Test);
                    if (Dead.getDeadCircleExist())
                    {
                        LOC t = CB.eatCBox(Player);
                        pace.emplace_back(t); //**记录步伐
                    }
                    else
                        break;
                }
                //然后开始考虑DoubleCross
                DCMove = CB.getDoubleCrossLoc(Player);
            }

            CB.move(Player, DCMove);
            pace.emplace_back(DCMove); //**记录步伐
            for (;;)                   //直到无法占据CTypeBox了就结束
            {
                if (!CB.getCTypeBoxLimit(Player, pace)) //**记录步伐
                    break;
            }
            //牺牲结束
        }
    }
    else
    {
        //选择打开哪一条长链。根据理性状态决定是打开最短的长链还是如何
        CB.eatAllCTypeBoxes(Player, pace); //**记录步伐
        BoxBoard Test(CB);

        LOC p = Test.openPolicy(); //调用打开决策函数获得打开步伐坐标
        CB.move(Player, p);
        pace.emplace_back(p); //**记录步伐
    }
}

//游戏移动，会根据前中后期自动移动
void gameTurnMove(Board &CB, int Player, volatile int *status, vector<LOC> &pace)
{
    // This Function is using for the game's move turn.

    // 游戏已结束，不再执行任何操作
    if (CB.ifEnd())
    {
        *status = 1;
        return;
    }

    if (USE_ALPHAZERO_AI)
    {
        // AlphaZero 路线：PUCT 搜索 + 精确终局求解器
        AlphaZeroMove(CB, Player, pace);
    }
    else
    {
        // 旧版 UCT 路线
        Board Test = CB;
        Test.eatAllCTypeBoxes(Player);
        bool LatterSituation = (Test.getFilterMoveNum() == 0);
        if (!LatterSituation)
            UCTMoveWithSacrifice(CB, Player, pace);
        else //也就是后期局面了
        {
            //也就是Filter都已经无能为力的情况下，只有LongChain,Circle,PreCircle
            latterSituationMove(CB, Player, pace); //**记录步伐
        }
    }
    *status = 1;
}

int rndFilterTurn(Board &CB, int Player)
{
    LOC Moves[60];
    CB.eatAllCTypeBoxes(Player); //此处第二个参数无实际意义

    BoxBoard Test = CB;
    int MoveNum;
    int FreeEdge = CB.getFreeEdgeNum();
    if (FreeEdge < UCT_FILTER_RANGE)          //仅在FreeEdge数量小于25的情况下考虑Filter（过滤）
        MoveNum = Test.getFilterMoves(Moves); //确定这个局面下MoveNum的数量
    else
        MoveNum = Test.getFreeMoves(Moves); //确定这个局面下MoveNum的数量

    if (MoveNum !=
        0) //在某些时候，由于吃掉了前面的C型格。可能导致MoveNum的数量为0.这时候只要跳过这一步自然就会开始判断胜利。
    {
        int Rnd = rng() % MoveNum; //在0-MoveNum中抽取一个随机数
        CB.move(Player, {Moves[Rnd].first, Moves[Rnd].second});
        return -Player; //换手
    }
    else
    {
        return Player; //不换手
    }
}