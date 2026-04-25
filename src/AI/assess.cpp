#include "assess.h"

BoxType::BoxType()
{
    Type = 0;              // 默认类型：未被占领
    BelongingChainNum = 0; // 默认为必空的0号空间
}

ChainInfo::ChainInfo()
{
    Type = NotDefine;             // 类型为未定义
    ChainBoxNum = 0;              // 所属格子数在开始时候为0
    ConditionOfPreCircle = false; // 是否预备环的先决条件
}

BoxBoard::BoxBoard(Board NewB) : Board(NewB)
{
}

BoxBoard::BoxBoard(int m[LEN][LEN]) : Board(m)
{
}

void BoxBoard::resetChainsInfo() // 重置链与格的信息，不使用0号空间
{
    for (int y = 1; y <= BOXLEN; y++)
    {
        for (int x = 1; x <= BOXLEN; x++)
        {
            Boxes[x][y].BelongingChainNum = 0; // 初始化格子所属链类型的编号
        }
    }
    for (int i = 1; i < 25; i++)
    {
        Chains[i].ChainBoxNum = 0;
        Chains[i].Type = NotDefine;
        Chains[i].ConditionOfPreCircle = false; // 预备环的先决条件
    }
}

int BoxBoard::getFilterMoves(LOC Moves[60])
{
    bool st[LEN][LEN] = {false};
    int MoveNum = 0;
    for (int x = 0; x < 11; x += 2)
    {
        for (int y = 1; y < 11; y += 2)
        {
            if (map[x][y] == HENG)
            {
                int BoardSave[LEN][LEN];
                boardCopy(map, BoardSave); // 保存一下
                move(BLACK, {x, y});       // 玩家模拟走一步试试
                if (x == 0)
                {
                    if (!getLongCTypeBoxBool(x + 1, y)) // 如果右边的那个格子没问题的话，这个招法也没问题
                    {
                        BoxBoard test(BoardSave);
                        if (test.getSaveChainEdgeBool(x + 1, y, st) && test.getSaveAngleEdgeBool(x + 1, y, st))
                        {
                            Moves[MoveNum] = {x, y};
                            MoveNum++; // 总数目自增
                        }
                        // else
                        //   cout<<"("<<x<<","<<y<<")"<<endl;
                    }
                }
                else if (x == LEN - 1)
                {
                    if (!getLongCTypeBoxBool(x - 1, y)) // 如果左边的那个格子没问题的话，这个招法也没问题
                    {
                        BoxBoard test(BoardSave);
                        if (test.getSaveChainEdgeBool(x - 1, y, st) && test.getSaveAngleEdgeBool(x - 1, y, st))
                        {
                            Moves[MoveNum] = {x, y};
                            MoveNum++; // 总数目自增
                        }
                        // else
                        //   cout<<"("<<x<<","<<y<<")"<<endl;
                    }
                }
                else
                {
                    if (!getLongCTypeBoxBool(x + 1, y) &&
                        !getLongCTypeBoxBool(x - 1, y)) // 如果左右两边的格子都没问题的话，这个招法也没问题
                    {
                        BoxBoard test(BoardSave);
                        if (test.getSaveChainEdgeBool(x + 1, y, st) && test.getSaveChainEdgeBool(x - 1, y, st))
                        {
                            Moves[MoveNum] = {x, y};
                            MoveNum++; // 总数目自增
                        }
                        // else
                        //   cout<<"("<<x<<","<<y<<")"<<endl;
                    }
                }
                setBoard(BoardSave); // 还原
                st[x][y] = true;
            }
        }
    }
    for (int x = 1; x < 11; x += 2)
    {
        for (int y = 0; y < 11; y += 2)
        {
            if (map[x][y] == SHU)
            {
                int BoardSave[LEN][LEN];
                boardCopy(map, BoardSave); // 保存一下
                move(BLACK, {x, y});       // 玩家模拟走一步试试
                if (y == 0)
                {
                    if (!getLongCTypeBoxBool(x, y + 1)) // 如果下面的那个格子没问题的话，这个招法也没问题
                    {
                        BoxBoard test(BoardSave);
                        if (test.getSaveChainEdgeBool(x, y + 1, st) && test.getSaveAngleEdgeBool(x, y + 1, st))
                        {
                            Moves[MoveNum] = {x, y};
                            MoveNum++; // 总数目自增
                        }
                        // else
                        //   cout<<"("<<x<<","<<y<<")"<<endl;
                    }
                }
                else if (y == LEN - 1)
                {
                    if (!getLongCTypeBoxBool(x, y - 1)) // 如果上面的那个格子没问题的话，这个招法也没问题
                    {
                        BoxBoard test(BoardSave);
                        if (test.getSaveChainEdgeBool(x, y - 1, st) && test.getSaveAngleEdgeBool(x, y - 1, st))
                        {
                            Moves[MoveNum] = {x, y};
                            MoveNum++; // 总数目自增
                        }
                        // else
                        //   cout<<"("<<x<<","<<y<<")"<<endl;
                    }
                }
                else
                {
                    if (!getLongCTypeBoxBool(x, y + 1) &&
                        !getLongCTypeBoxBool(x, y - 1)) // 如果上下的格子都没问题的话，这个招法也没问题
                    {
                        BoxBoard test(BoardSave);
                        if (test.getSaveChainEdgeBool(x, y + 1, st) && test.getSaveChainEdgeBool(x, y - 1, st))
                        {
                            Moves[MoveNum] = {x, y};
                            MoveNum++; // 总数目自增
                        }
                        // else
                        //   cout<<"("<<x<<","<<y<<")"<<endl;
                    }
                }
                setBoard(BoardSave); // 还原
                st[x][y] = true;
            }
        }
    }
    // system("pause");
    return MoveNum;
}

void BoxBoard::searchingFromBox(LOC BoxLoc) // 从一个格出发，注册他的所有派生链，ChainPlus应在没有短链时启用
{
    // cout<<"进入searchingFromBox函数"<<endl;
    for (auto n : Dir)
    {
        int bx = BoxLoc.first + n[0];
        int by = BoxLoc.second + n[1]; //#邻格#
        int lx = (BoxLoc.first * 2) - 1 + n[0];
        int ly = (BoxLoc.second * 2) - 1 + n[1];                                           //#邻边#
        if ((map[lx][ly] == HENG || map[lx][ly] == SHU) && getBoxType(bx, by) == CHAINBOX) // 邻边为空，目标格子为链格
        {
            if (Boxes[bx][by].BelongingChainNum == EMPTY) // 必须为未归属的格子，避免环重复从不同方向出发。
            {
                LOC NewLoc;
                NewLoc = {bx, by};
                registerChain(BoxLoc, NewLoc);
            }
        }
    }
    // cout<<"完成searchingFromBox函数"<<endl;
}

int BoxBoard::getFirstEmptyChainNum() // 获取可用链空间的编号
{
    for (int i = 1; i < 25; i++) // 不采用0号空间
    {
        if (Chains[i].Type == NotDefine)
            return i;
    }
    // cout << "链空间不足！" << endl;
    // system("pause");
    return 0;
}

void BoxBoard::registerChain(LOC FreeBoxLoc, LOC FirstLoc)
{
    // 动态注册链。从一个格子出发 ，向一个格子开始进行链的注册。一般仅从自由格出发

    LOC Loc = FreeBoxLoc;             // 设置起点坐标
    LOC NewLoc = FirstLoc;            // 设置新坐标#目标格子#
    int Re = getFirstEmptyChainNum(); // 获取一个空白的链空间
    Chains[Re].StartLoc = Loc;        // 起点为某个自由格
    Chains[Re].ChainBoxNum = 0;       // 更新起点
    Chains[Re].Type = NotDefine;

    // 在如下过程中，NewLoc才是实际链的每一个格子的坐标。链的类型在搜索结束时候判断
    for (int i = 0; i < BOXNUM; i++) //
    {
        if (getBoxType(NewLoc.first, NewLoc.second) !=
            CHAINBOX) // 当搜索结束的时候，判定链的类型。此时，NewLoc为搜索到的终点格坐标
        {
            //#目标格子不为链格#
            Chains[Re].EndLoc = NewLoc; // 搜索到达终点
            if (NewLoc.first == FreeBoxLoc.first && NewLoc.second == FreeBoxLoc.second &&
                Boxes[NewLoc.first][NewLoc.second].BelongingChainNum == EMPTY)
            {
                // 若终点格的坐标与起点格相同并且该格还未被认定类型，则认定为PreCircle
                Chains[Re].Type = PreCircle;
                Boxes[NewLoc.first][NewLoc.second].BelongingChainNum = Re; // 包括该格在内也需要被注册
                Chains[Re].ChainBoxNum++;
            }
            else // 若起终点不一样，则为链。根据数目判定其类型
            {
                if (Chains[Re].ChainBoxNum <= 2)
                    Chains[Re].Type = ShortChain;
                else
                    Chains[Re].Type = LongChain;
            }
            break;
        }
        else // 每搜到一个新的格子
        {
            LOC mLoc = NewLoc;
            // 定义格子的信息
            Boxes[NewLoc.first][NewLoc.second].BelongingChainNum = Re; // 与链挂钩
            Chains[Re].ChainBoxNum++;                                  // 链的格子数目自增1
            NewLoc = findNextBox(NewLoc, Loc); // 寻找下一个。#参数为起始格子，目标格子#
            Loc = mLoc;
        }
    }
}

LOC BoxBoard::findNextBox(LOC FoundBox, LOC LastBox)
{
    // 查找某个链格,给出的坐标可能包含超区坐标。请使用前利用GetBoxType鉴定
    int fbx = FoundBox.first * 2 - 1;
    int fby = FoundBox.second * 2 - 1; //#转换成board坐标#
    LOC AimBox;
    if (map[fbx + 1][fby] != OCCLINE && ((FoundBox.first + 1) != LastBox.first || (FoundBox.second) != LastBox.second))
    {
        // 0度  FoundBox右邻边为空，同时右目标格子不是上上个找到的格子LastBox
        AimBox = {FoundBox.first + 1, FoundBox.second}; //#确定目标格子#
        return AimBox;
    }
    else if (map[fbx][fby + 1] != OCCLINE &&
             ((FoundBox.first) != LastBox.first || (FoundBox.second + 1) != LastBox.second))
    {
        // 90度  FoundBox下邻边为空，同时下目标格子不是上上个找到的格子LastBox
        AimBox = {FoundBox.first, FoundBox.second + 1};
        return AimBox;
    }
    else if (map[fbx - 1][fby] != OCCLINE &&
             ((FoundBox.first - 1) != LastBox.first || (FoundBox.second) != LastBox.second))
    {
        // 180度  FoundBox左邻边为空，同时左目标格子不是上上个找到的格子LastBox
        AimBox = {FoundBox.first - 1, FoundBox.second};
        return AimBox;
    }
    else if (map[fbx][fby - 1] != OCCLINE &&
             ((FoundBox.first) != LastBox.first || (FoundBox.second - 1) != LastBox.second))
    {
        // 270度  FoundBox上邻边为空，同时上目标格子不是上上个找到的格子
        AimBox = {FoundBox.first, FoundBox.second - 1};
        return AimBox;
    }
    AimBox = {0, 0}; // 代表错误
    return AimBox;
}

int BoxBoard::getBoxType(int bx, int by) // 从棋盘上获取格子类型
{
    // cout << "进入getBoxType" << endl;
    if (bx >= 1 && by >= 1 && bx <= BOXLEN && by <= BOXLEN) // 满足格子要求
        return Boxes[bx][by].Type;
    return FREEBOX; // 其余超出范围的均返回自由格
    // cout << "完成getBoxType" << endl;
}

void BoxBoard::defineAllChains(bool ChainPlus) // 定义所有的链
{
    // cout << "进入defineAllChains" << endl;
    defineBoxesType(); // 首先定义所有格子的类型
    resetChainsInfo(); // 重置链的数据

    // 首先从自由格出发搜索所有长链
    for (int i = 1; i <= BOXLEN; i++)
    {
        for (int j = 1; j <= BOXLEN; j++)
        {
            if (Boxes[i][j].Type == 3)
            {
                LOC k;
                k = {i, j};
                searchingFromBox(k);
            }
        }
    }

    // 然后在边缘搜索还没有被排查的链
    for (int i = 1; i <= BOXLEN; i++)
    {
        LOC k;
        k = {i, 0};
        searchingFromBox(k);
    }
    for (int i = 1; i <= BOXLEN; i++)
    {
        LOC k;
        k = {0, i};
        searchingFromBox(k);
    }
    for (int i = 1; i <= BOXLEN; i++)
    {
        LOC k;
        k = {i, BOXLEN + 1};
        searchingFromBox(k);
    }
    for (int i = 1; i <= BOXLEN; i++)
    {
        LOC k;
        k = {BOXLEN + 1, i};
        searchingFromBox(k);
    }

    // 然后搜索还没有被搜索到的链格，可能为环
    for (int i = 1; i <= BOXLEN; i++)
    {
        for (int j = 1; j <= BOXLEN; j++)
        {
            if (Boxes[i][j].Type == CHAINBOX && Boxes[i][j].BelongingChainNum == EMPTY)
            {
                LOC BoxLoc;
                BoxLoc = {i, j};
                searchingCircle(BoxLoc);
            }
        }
    }

    // 暂时没用到，定义与预备链和合并自由格相连的两条链
    // 最后开始ChainPlus
    if (ChainPlus)
    {
        // 首先查找是否有首尾相同的链，定义为PreChain
        for (int i = 1; i < 25; i++)
        {
            if (Chains[i].Type == LongChain)
            {
                for (int j = i + 1; j <= 25; j++)
                {
                    if (Chains[j].Type == LongChain && Chains[i].StartLoc == Chains[j].StartLoc &&
                        Chains[i].EndLoc == Chains[j].EndLoc)
                    {
                        if (Boxes[Chains[j].StartLoc.first][Chains[j].StartLoc.second].BelongingChainNum == EMPTY &&
                            Boxes[Chains[j].EndLoc.first][Chains[j].EndLoc.second].BelongingChainNum == EMPTY)
                        {
                            // 此时，i与j同为长链而且首尾相同且首尾的格子都未被认定，可以认定他们为PreChain
                            inheritChain(i, j); // 链i继承了链j
                            Boxes[Chains[i].StartLoc.first][Chains[i].StartLoc.second].BelongingChainNum = i;
                            Boxes[Chains[i].EndLoc.first][Chains[i].EndLoc.second].BelongingChainNum = i;
                            Chains[i].ChainBoxNum = Chains[i].ChainBoxNum + 2;
                            Chains[i].StartLoc = Chains[i].EndLoc;
                            Chains[i].Type = PreCircle;
                        }
                    }
                }
            }
        }

        // 然后把所有PreCircle的PreChain定义好
        for (int y = 1; y <= BOXLEN; y++)
        {
            for (int x = 1; x <= BOXLEN; x++)
            {
                if (Boxes[x][y].Type == 3 && Chains[Boxes[x][y].BelongingChainNum].Type == PreCircle)
                {
                    int bx = x * 2 - 1;
                    int by = y * 2 - 1;
                    // 先手先产生一个死格
                    for (auto n : Dir)
                    {
                        int ex = bx + n[0];
                        int ey = by + n[1]; //#邻边#
                        int nbx = x + n[0];
                        int nby = y + n[1]; //#邻格#
                        if (map[ex][ey] != OCCLINE && Chains[Boxes[nbx][nby].BelongingChainNum].Type == LongChain)
                        {
                            Chains[Boxes[nbx][nby].BelongingChainNum].ConditionOfPreCircle = true;
                        }
                    }
                }
            }
        }

        // 最后把所有还没定义的自由格归属到他最长的两条派生链的合并中
        for (int y = 1; y <= BOXLEN; y++)
        {
            for (int x = 1; x <= BOXLEN; x++)
            {
                if (Boxes[x][y].Type == FREEBOX && Boxes[x][y].BelongingChainNum == EMPTY) // 若本自由格还未确定
                {
                    int ChainRegNum[4];
                    int Total = 0;
                    for (auto n : Dir)
                    {
                        int bx = x + n[0];
                        int by = y + n[1];
                        int lx = (x * 2) - 1 + n[0];
                        int ly = (y * 2) - 1 + n[1];
                        if (map[lx][ly] != OCCLINE && getBoxType(bx, by) == CHAINBOX) // 邻边为空，目标格子为链格
                        {
                            ChainRegNum[Total] = Boxes[bx][by].BelongingChainNum;
                            Total++;
                        }
                    }
                    // 先找出最长的
                    int FirstChainNum = 0;
                    int FirstChainBoxes = 0;
                    for (int i = 0; i < Total; i++)
                    {
                        if (Chains[ChainRegNum[i]].ChainBoxNum > FirstChainBoxes)
                        {
                            FirstChainNum = ChainRegNum[i];
                            FirstChainBoxes = Chains[ChainRegNum[i]].ChainBoxNum;
                        }
                    }
                    // 再找出次长的
                    int SecChainNum = 0;
                    int SecChainBoxes = 0;
                    for (int i = 0; i < Total; i++)
                    {
                        if (Chains[ChainRegNum[i]].ChainBoxNum > SecChainBoxes && ChainRegNum[i] != FirstChainNum)
                        {
                            SecChainNum = ChainRegNum[i];
                            SecChainBoxes = Chains[ChainRegNum[i]].ChainBoxNum;
                        }
                    }
                    // 将两个链，以及本格累加在一起
                    Boxes[x][y].BelongingChainNum = FirstChainNum;
                    Chains[FirstChainNum].ChainBoxNum++;
                    inheritChain(FirstChainNum, SecChainNum);
                }
            }
        }
    }
}

void BoxBoard::searchingCircle(LOC BoxLoc)
{
    for (auto n : Dir)
    {
        int bx = BoxLoc.first + n[0];
        int by = BoxLoc.second + n[1];
        int lx = (BoxLoc.first * 2) - 1 + n[0];
        int ly = (BoxLoc.second * 2) - 1 + n[1];
        if (map[lx][ly] != OCCLINE && getBoxType(bx, by) == CHAINBOX) // 邻边为空，目标格子为链格
        {
            if (Boxes[bx][by].BelongingChainNum == EMPTY) // 必须为未归属的格子，避免环重复从不同方向出发。
            {
                LOC NewLoc;
                NewLoc = {bx, by};
                registerCircle(BoxLoc, NewLoc);
            }
        }
    }
}

void BoxBoard::registerCircle(LOC StartLoc, LOC NextLoc) // 动态注册链.一般仅从自由格出发
{
    // cout << "进入registerCircle函数" << endl;
    LOC Loc = StartLoc;               // 设置起点坐标
    LOC NewLoc = NextLoc;             // 设置新坐标
    int Re = getFirstEmptyChainNum(); // 获取一个空白的链空间
    Chains[Re].ChainBoxNum = 0;       // 改为0
    Chains[Re].Type = NotDefine;
    // 在如下过程中，NewLoc才是实际链的每一个格子的坐标。链的类型在搜索结束时候判断
    for (int i = 0; i < 25; i++)
    {
        if (Boxes[NewLoc.first][NewLoc.second].Type != 2) // 若非链格，则取消注册。
        {
            inheritChain(EMPTY, Re); // 0号链吞并所有已经注册的链
            break;
        }
        else if (NewLoc.first == StartLoc.first && NewLoc.second == StartLoc.second) // 搜到起点，确定为Circle
        {
            Chains[Re].Type = Circle;
            Chains[Re].StartLoc = NewLoc;                              // 起点与终点都标记为这个格子
            Chains[Re].EndLoc = NewLoc;                                // 起点与终点都标记为这个格子
            Boxes[NewLoc.first][NewLoc.second].BelongingChainNum = Re; // 包括该格在内也需要被注册
            Chains[Re].ChainBoxNum++;                                  // 链的格子数目自增1
            break;
        }
        else // 每搜到一个新的格子
        {
            LOC mLoc = NewLoc;
            // 定义格子的信息
            Boxes[NewLoc.first][NewLoc.second].BelongingChainNum = Re; // 与链挂钩
            Chains[Re].ChainBoxNum++;                                  // 链的格子数目自增1
            NewLoc = findNextBox(NewLoc, Loc);                         // 寻找下一个。
            Loc = mLoc;
        }
    }
    // cout << "完成registerCircle函数" << endl;
}

void BoxBoard::inheritChain(int InheritorRegNum, int AncesterRegNum)
{
    // cout << "进入inheritChain函数" << endl;
    for (int j = 1; j <= BOXLEN; j++)
    {
        for (int i = 1; i <= BOXLEN; i++)
        {
            if (Boxes[i][j].BelongingChainNum == AncesterRegNum)
                Boxes[i][j].BelongingChainNum = InheritorRegNum; // 被继承
        }
    }
    Chains[InheritorRegNum].ChainBoxNum += Chains[AncesterRegNum].ChainBoxNum; // 格子数累加
    Chains[InheritorRegNum].Type = LongChain;                                  // 必定是长链。

    // 销毁被继承者的注册空间
    Chains[AncesterRegNum].ChainBoxNum = 0;
    Chains[AncesterRegNum].Type = NotDefine;
    // cout << "完成inheritChain函数" << endl;
}

void BoxBoard::defineDeadChain()
{
    // cout << "进入defineDeadChain函数" << endl;
    defineBoxesType(); // 首先定义所有格子的类型
    resetChainsInfo(); // 重置链的数据
    for (int i = 1; i <= BOXLEN; i++)
    {
        for (int j = 1; j <= BOXLEN; j++)
        {
            if (Boxes[i][j].Type == DEADBOX)
            {
                LOC k;
                k = {i, j};
                searchingDeadChain(k);
            }
        }
    }
    // cout << "完成defineDeadChain函数" << endl;
}

void BoxBoard::searchingDeadChain(LOC BoxLoc)
{
    // cout << "进入searchingDeadChain" << endl;
    for (auto n : Dir)
    {
        int bx = BoxLoc.first + n[0];
        int by = BoxLoc.second + n[1];
        int lx = (BoxLoc.first * 2) - 1 + n[0];
        int ly = (BoxLoc.second * 2) - 1 + n[1];
        if ((map[lx][ly] == HENG || map[lx][ly] == SHU) && getBoxType(bx, by) == CHAINBOX) // 邻边为空，目标格子为链格
        {
            if (Boxes[bx][by].BelongingChainNum == EMPTY) // 必须为未归属的格子，避免环重复从不同方向出发。
            {
                LOC NewLoc;
                NewLoc = {bx, by};
                registerDeadChain(BoxLoc, NewLoc);
            }
        }
    }
    // cout << "完成searchingDeadChain" << endl;
}

void BoxBoard::registerDeadChain(LOC FreeBoxLoc, LOC FirstLoc)
{
    // 动态注册链。从一个格子出发 ，向一个格子开始进行链的注册。
    //  cout << "进入registerDeadChain" << endl;

    LOC Loc = FreeBoxLoc;             // 设置起点坐标
    LOC NewLoc = FirstLoc;            // 设置新坐标
    int Re = getFirstEmptyChainNum(); // 获取一个空白的链空间
    Chains[Re].StartLoc = Loc;        // 起点为某个自由格
    Chains[Re].ChainBoxNum = 1;       // 更新起点
    Chains[Re].Type = NotDefine;

    // 在如下过程中，NewLoc才是实际链的每一个格子的坐标。链的类型在搜索结束时候判断
    for (int i = 0; i < BOXNUM; i++)
    {
        if (getBoxType(NewLoc.first, NewLoc.second) !=
            CHAINBOX) // 当搜索结束的时候，判定链的类型。此时，NewLoc为搜索到的终点格坐标
        {
            Chains[Re].EndLoc = NewLoc; // 搜索到达终点
            if (getBoxType(NewLoc.first, NewLoc.second) ==
                DEADBOX) // 如果终端是一个死格的话，这就是一个死循环(DeadCircle)
            {
                Chains[Re].Type = DeadCircle;
                Boxes[NewLoc.first][NewLoc.second].BelongingChainNum = Re; // 包括该格在内也需要被注册
                Chains[Re].ChainBoxNum++;
            }
            else // 若起终点不一样，则为链。根据数目判定其类型
            {
                Chains[Re].Type = DeadChain;
            }
            break;
        }
        else // 每搜到一个新的格子
        {
            LOC mLoc = NewLoc;
            // 定义格子的信息
            Boxes[NewLoc.first][NewLoc.second].BelongingChainNum = Re; // 与链挂钩
            Chains[Re].ChainBoxNum++;                                  // 链的格子数目自增1
            NewLoc = findNextBox(NewLoc, Loc);                         // 寻找下一个。
            Loc = mLoc;
        }
    }

    // cout << "完成registerDeadChain" << endl;
}

bool BoxBoard::captualShortestChain(int LatterPlayer)
{
    defineAllChains(false);

    // 先找到最短的那条长链/环
    int Least = 0;
    int LeastBoxNum = BOXNUM; // 假设最短的链有25个
    for (int i = 1; i < BOXNUM; i++)
    {
        if (Chains[i].Type == LongChain || Chains[i].Type == Circle)
        {
            if (Chains[i].ChainBoxNum < LeastBoxNum)
            {
                LeastBoxNum = Chains[i].ChainBoxNum;
                Least = i;
            }
        }
    }
    if (Least == 0)
        return false;

    // 先手先占领任意两个格子之间的边
    bool Finish = false;
    for (int y = 1; y <= BOXLEN; y++)
    {
        for (int x = 1; x <= BOXLEN; x++)
        {
            if (Boxes[x][y].BelongingChainNum == Least && !Finish)
            {
                int bx = x * 2 - 1;
                int by = y * 2 - 1;

                for (auto n : Dir)
                {
                    int ex = bx + n[0];
                    int ey = by + n[1];
                    int nbx = x + n[0];
                    int nby = y + n[1];
                    if ((map[ex][ey] == HENG || map[ex][ey] == SHU) &&
                        Boxes[nbx][nby].BelongingChainNum == Least) // 找到两个格子交叉的那个格子
                    {
                        move(-LatterPlayer, {ex, ey});
                        Finish = true;
                        break;
                    }
                }
            }
        }
    }

    // 后手占领这条链所有其他格子空余的边
    for (int y = 1; y <= BOXLEN; y++)
    {
        for (int x = 1; x <= BOXLEN; x++)
        {
            if (Boxes[x][y].BelongingChainNum == Least)
            {
                int bx = x * 2 - 1;
                int by = y * 2 - 1;
                for (auto n : Dir)
                {
                    int ex = bx + n[0];
                    int ey = by + n[1];
                    if (map[ex][ey] == HENG || map[ex][ey] == SHU)
                    {
                        move(LatterPlayer, {ex, ey});
                        break;
                    }
                }
                // 占据结束，刷新链信息，结束。
                Boxes[x][y].BelongingChainNum = 0;
            }
        }
    }
    // 结束
    inheritChain(EMPTY, Least);
    defineBoxesType();
    return true;
}

LOC BoxBoard::getOpenShortestChainLoc()
{
    defineAllChains(false);

    // 先找到最短的那条长链/环
    int Least = 0;
    int LeastBoxNum = BOXNUM; // 假设最短的链有25个
    for (int i = 1; i < BOXNUM; i++)
    {
        if (Chains[i].Type == LongChain)
        {
            if (Chains[i].ChainBoxNum < LeastBoxNum)
            {
                LeastBoxNum = Chains[i].ChainBoxNum;
                Least = i;
            }
        }
    }

    // 先手先占领任意两个格子之间的边
    for (int y = 1; y <= BOXLEN; y++)
    {
        for (int x = 1; x <= BOXLEN; x++)
        {
            if (Boxes[x][y].BelongingChainNum == Least)
            {
                int bx = x * 2 - 1;
                int by = y * 2 - 1;

                for (auto n : Dir)
                {
                    int ex = bx + n[0];
                    int ey = by + n[1];
                    int nbx = x + n[0];
                    int nby = y + n[1];
                    if ((map[ex][ey] == HENG || map[ex][ey] == SHU) &&
                        Boxes[nbx][nby].BelongingChainNum == Least) // 找到两个格子交叉的那个格子
                    {
                        LOC k;
                        k = {ex, ey};
                        return k;
                    }
                }
            }
        }
    }
    LOC k;
    k = {0, 0};
    return k;
}

LOC BoxBoard::openPolicy() // 打开策略
{

    LOC res = {-1, -1};

    defineAllChains(false);
    int LCnum = 0;  // 长链
    int Cnum = 0;   // 环
    int LC3num = 0; // 3格长链
    for (int i = 1; i < 25; i++)
    {
        if (Chains[i].Type == Circle)
            Cnum++;
        if (Chains[i].Type == LongChain)
        {
            LCnum++;
            if (Chains[i].ChainBoxNum == 3)
                LC3num++;
        }
    }
    // cout << "LCnum:" << LCnum << " "
    //      << "Cnum:" << Cnum << " "
    //      << "LC3num:" << LC3num << endl;
    // 打开策略
    if (LCnum == 0 && Cnum != 0) // 如果G不含长链，应开启最短的环；
        res = getOpenShortestCircleLoc();
    else if (LCnum != 0 && Cnum == 0) // 如果G不含环，应开启最短的链
        res = getOpenShortestChainLoc();
    else if (LCnum != 0 && Cnum != 0 && LC3num == 0) // 如果G含环但是不包含3-链，应开启最短的环
        res = getOpenShortestCircleLoc();
    else if (LCnum != 0 && Cnum != 0 && LC3num != 0)
        res = getOpen3ChainLoc();

    return res;
}

LOC BoxBoard::getOpenShortestCircleLoc() // 获得待打开的最短的环的坐标
{
    defineAllChains(false);

    // 先找到最短的那条环
    int Least = 0;
    int LeastBoxNum = BOXNUM; // 假设最短的环长为25
    for (int i = 1; i < BOXNUM; i++)
    {
        if (Chains[i].Type == Circle)
        {
            if (Chains[i].ChainBoxNum < LeastBoxNum)
            {
                LeastBoxNum = Chains[i].ChainBoxNum;
                Least = i;
            }
        }
    }

    // 先手先占领任意两个格子之间的边
    for (int y = 1; y <= BOXLEN; y++)
    {
        for (int x = 1; x <= BOXLEN; x++)
        {
            if (Boxes[x][y].BelongingChainNum == Least)
            {
                int bx = x * 2 - 1;
                int by = y * 2 - 1;

                for (auto n : Dir)
                {
                    int ex = bx + n[0];
                    int ey = by + n[1];
                    int nbx = x + n[0];
                    int nby = y + n[1];
                    if ((map[ex][ey] == HENG || map[ex][ey] == SHU) &&
                        Boxes[nbx][nby].BelongingChainNum == Least) // 找到两个格子共用的那条边
                    {
                        LOC k;
                        k = {ex, ey};
                        return k;
                    }
                }
            }
        }
    }
    LOC k;
    k = {0, 0};
    return k;
}

LOC BoxBoard::getOpen3ChainLoc() // 获得打开3链的坐标
{
    defineAllChains(false);
    // 先找到要打开的结构体的
    int SuitChainidx = 60;           // 编号
    for (int i = 1; i < BOXNUM; i++) // 有3链先打开3链
    {
        if (Chains[i].Type == LongChain && Chains[i].ChainBoxNum == 3)
        {
            SuitChainidx = i;
        }
    }

    // 获得要打开的结构体中可占坐标
    for (int y = 1; y <= BOXLEN; y++)
    {
        for (int x = 1; x <= BOXLEN; x++)
        {
            if (Boxes[x][y].BelongingChainNum == SuitChainidx)
            {
                int bx = x * 2 - 1;
                int by = y * 2 - 1;

                for (auto n : Dir)
                {
                    int ex = bx + n[0];
                    int ey = by + n[1];
                    int nbx = x + n[0];
                    int nby = y + n[1];
                    if ((map[ex][ey] == HENG || map[ex][ey] == SHU) &&
                        Boxes[nbx][nby].BelongingChainNum == SuitChainidx) // 找到两个格子交叉的那条边
                    {
                        LOC k;
                        k = {ex, ey};
                        return k;
                    }
                }
            }
        }
    }
    LOC k;
    k = {0, 0};
    return k;
}

bool BoxBoard::rationalState(LOC BoxNum)
{
    if (BoxNum.first >= BoxNum.second)
        return true;
    return false;
}

LOC BoxBoard::getEarlyRationalBoxNum() // 用于UCT预处理，获得余下局面双方以理性状态可以获得的格子数量
{
    defineAllChains(false);

    // 首先清算各种链的数目
    int LCNum = 0; // 长链数目
    int LCBox = 0;
    int CNum = 0; // 环数目
    int CBox = 0;
    int PCBox = 0;

    for (int i = 1; i <= 25; i++)
    {
        if (Chains[i].Type != NotDefine)
        {
            if (Chains[i].Type == LongChain)
            {
                LCNum++;
                LCBox += Chains[i].ChainBoxNum;
            }
            else if (Chains[i].Type == Circle)
            {
                CNum++;
                CBox += Chains[i].ChainBoxNum;
            }
        }
    }

    // 开始计算理性状态下，对方牺牲的格子数(也就是对方让给我方的格子数)
    int Total = LCBox + PCBox + CBox;
    int Sacrifice = 0;
    if (LCNum == 0 && CNum != 0) // 此状况下不存在长链，则都是环
    {
        Sacrifice = (CNum - 1) * 4; // 最后一个环对方肯定会吃掉，所以要-1
    }
    if (LCNum != 0)
    {
        // 有长链的时候，最后一个必定是长链
        Sacrifice = (CNum * 4) + (LCNum * 2) - 2;
    }
    LOC num;
    num = {Total - Sacrifice, Sacrifice};
    return num;
}

LOC BoxBoard::getRationalStateBoxNum()
{
    defineAllChains(true);

    // 首先清算各种链的数目
    int LCNum = 0; // 长链
    int LCBox = 0;
    int CNum = 0; // 环
    int CBox = 0;
    int PCNum = 0; // 预备环
    int PCBox = 0;
    bool OnlyPreChain = true; // 是否仅有预备链，也就是预备环的先决条件。是的话，最后一个必定是预备环

    for (int i = 1; i <= BOXNUM; i++)
    {
        if (Chains[i].Type != NotDefine)
        {
            if (Chains[i].Type == LongChain)
            {
                if (Chains[i].ConditionOfPreCircle == false)
                    OnlyPreChain = false;
                LCNum++;
                LCBox += Chains[i].ChainBoxNum;
            }
            else if (Chains[i].Type == Circle)
            {
                CNum++;
                CBox += Chains[i].ChainBoxNum;
            }
            else if (Chains[i].Type == PreCircle)
            {
                PCNum++;
                PCBox += Chains[i].ChainBoxNum;
            }
        }
    }

    // 开始计算牺牲数目
    int Total = LCBox + PCBox + CBox; // 总格子数
    int Sacrifice = 0;
    if (OnlyPreChain)
    {
        if (LCNum == 0) // 但这个状况下不存在长链，则该情况都是环
        {
            Sacrifice = (CNum - 1) * 4;
        }
        else // 存在长链，但都是预备链。最后一个必定为预备环
        {
            Sacrifice = (PCNum * 4) + (CNum * 4) + (LCNum * 2) - 4;
        }
    }
    else
    {
        // 有长链的时候，最后一个必定是长链
        Sacrifice = (PCNum * 4) + (CNum * 4) + (LCNum * 2) - 2;
    }

    return {Total - Sacrifice, Sacrifice};
}

int BoxBoard::getBoardWinner(int LatterPlayer)
{
    // if (getFilterMoveNum() > 0)
    //     cout << "Wrong";
    int player = LatterPlayer;
    // defineBoxesType();

    LOC BoxNum;
    for (;;) // 非理性情况下吞并所有
    {
        // defineAllChains(true);             // 先定义为完全状态判定一次
        BoxNum = getRationalStateBoxNum(); // 然后再判定一次理性情况
        if (rationalState(BoxNum))
            break;
        else
        {
            if (!captualShortestChain(player)) // 如果吃不下去了也退出
                break;
            else
                player = -player;
        }
    }
    if (getWinner() == 0) // 也就是还没胜利
    {
        int r, b;
        if (player == BLACK)
        {
            r = BoxNum.first + getPlayerBoxes(BLACK); // 玩家是红方，则加上除去牺牲剩余的格子数
            b = BoxNum.second + getPlayerBoxes(WHITE);
        }
        else
        {
            r = BoxNum.second + getPlayerBoxes(BLACK);
            b = BoxNum.first + getPlayerBoxes(WHITE); // 玩家是蓝方，则加上牺牲剩余的格子数
        }
        if (r > b)
            return BLACK;
        else
            return WHITE;
    }
    else
        return getWinner();
}

int BoxBoard::getBoardWinner(int LatterPlayer, int FaOwner, int &score)
{
    // if (getFilterMoveNum() > 0)
    //     cout << "Wrong";
    int player = LatterPlayer;
    // defineBoxesType();

    LOC BoxNum;
    for (;;) // 非理性情况下吞并所有
    {
        // defineAllChains(true);             // 先定义为完全状态判定一次
        BoxNum = getRationalStateBoxNum(); // 然后再判定一次理性情况
        if (rationalState(BoxNum))
            break;
        else
        {
            if (!captualShortestChain(player)) // 如果吃不下去了也退出
                break;
            else
                player = -player;
        }
    }
    if (getWinner() == 0) // 也就是还没下完
    {
        int r, b;
        if (player == BLACK)
        {
            r = BoxNum.first + getPlayerBoxes(BLACK); // 玩家是红方，则加上除去牺牲剩余的格子数
            b = BoxNum.second + getPlayerBoxes(WHITE);
            if (FaOwner == BLACK)
                score = r; // score为该节点的父结点拥有者即player获得的格子数
            else
                score = b;
        }
        else
        {
            r = BoxNum.second + getPlayerBoxes(BLACK);
            b = BoxNum.first + getPlayerBoxes(WHITE); // 玩家是蓝方，则加上牺牲剩余的格子数
            if (FaOwner == BLACK)
                score = r; // score为该节点的父结点拥有者即player获得的格子数
            else
                score = b;
        }
        if (r > b)
            return BLACK;
        else
            return WHITE;
    }
    else
    {
        if (FaOwner == BLACK)
            score = blackBox; // score为该节点的父结点拥有者即player获得的格子数
        else
            score = whiteBox;
        return getWinner();
    }
}

bool BoxBoard::getDeadChainExist()
{
    defineDeadChain();
    for (int i = 0; i < BOXNUM; i++)
    {
        if (Chains[i].Type == DeadChain)
            return true;
    }
    return false;
}
bool BoxBoard::getDeadCircleExist()
{
    defineDeadChain();
    for (int i = 0; i < BOXNUM; i++)
    {
        if (Chains[i].Type == DeadCircle && Chains[i].ChainBoxNum > 3) // 保证留两个双交
            return true;
    }
    return false;
}

void BoxBoard::defineBoxesType() // 定义格子类型
{
    for (int y = 1; y <= BOXLEN; y++) // 格子数
    {
        for (int x = 1; x <= BOXLEN; x++) // 格子数
        {
            int bx = (x * 2) - 1; // Board中的坐标#表格子属主坐标#
            int by = (y * 2) - 1; // Board中的坐标
            // 定义Type
            int bl = getFreedom(bx, by); // 得到格子的自由度
            if (bl == 4)
                Boxes[x][y].Type = FREEBOX; // 是自由格
            else
                Boxes[x][y].Type = bl; // 不然就跟自由边的数量是相同的。
        }
    }
}

bool BoxBoard::getSaveChainEdgeBool(int x, int y, bool st[LEN][LEN]) // 用于获得过滤可行边时，判断是否存下该条边
{
    if (!isOdd(x) || !isOdd(y))
    {
        //   cout<<"判断是否要存边时:传入错误的坐标信息!"<<endl;
        return false;
    }
    defineAllChains(false);
    //    for(int i=01;i<25;i++)
    //    {
    // 	 if(Chains[i].Type==SingleChain)
    // 	 {
    //         cout<<i<<endl;
    // 	    cout<<Chains[i].StartLoc.x<<","<<Chains[i].StartLoc.y<<"到"<<Chains[i].EndLoc.x<<","<<Chains[i].EndLoc.y<<endl;
    // 	 }
    //    }
    //    cout<<"  "<<endl;

    int idx = Boxes[(x + 1) / 2][(y + 1) / 2].BelongingChainNum;
    if (Chains[idx].Type == ShortChain && Chains[idx].ChainBoxNum == 1) // 裁剪一格短链的等价边
    {
        int cnt = 0;
        if ((map[x][y - 1] == HENG || map[x][y - 1] == SHU) && st[x][y - 1] == false)
            cnt++;
        if ((map[x + 1][y] == HENG || map[x + 1][y] == SHU) && st[x + 1][y] == false)
            cnt++;
        if ((map[x][y + 1] == HENG || map[x][y + 1] == SHU) && st[x][y + 1] == false)
            cnt++;
        if ((map[x - 1][y] == HENG || map[x - 1][y] == SHU) && st[x - 1][y] == false)
            cnt++;

        if (cnt == 2)
            return true;
        if (cnt == 1)
            return false;
    }

    return true; // 若不是1格短链，无需等价裁边
}

bool BoxBoard::getSaveAngleEdgeBool(int x, int y, bool st[LEN][LEN]) // 对边角格子等价裁边
{
    int cnt = 0;
    if (x == 1 && y == 1)
    {
        if (getFreedom(x, y) == 4)
        {
            if (st[x][y - 1] == false)
                cnt++;
            if (st[x - 1][y] == false)
                cnt++;
            if (cnt == 2)
                return true;
            if (cnt == 1)
                return false;
        }
        return true;
    }
    if (x == 9 && y == 1)
    {
        if (getFreedom(x, y) == 4)
        {
            if (st[x][y - 1] == false)
                cnt++;
            if (st[x + 1][y] == false)
                cnt++;
            if (cnt == 2)
                return true;
            if (cnt == 1)
                return false;
        }
        return true;
    }
    if (x == 1 && y == 9)
    {
        if (getFreedom(x, y) == 4)
        {
            if (st[x - 1][y] == false)
                cnt++;
            if (st[x][y + 1] == false)
                cnt++;
            if (cnt == 2)
                return true;
            if (cnt == 1)
                return false;
        }
        return true;
    }
    if (x == 9 && y == 9)
    {
        if (getFreedom(x, y) == 4)
        {
            if (st[x][y + 1] == false)
                cnt++;
            if (st[x + 1][y] == false)
                cnt++;
            if (cnt == 2)
                return true;
            if (cnt == 1)
                return false;
        }
        return true;
    }
    return true;
}

int BoxBoard::getFreeMoves(LOC Moves[60])
{
    bool st[LEN][LEN] = {false};
    int MoveNum = 0;
    // 得到所有的自由边
    // 先判断所有竖边
    for (int y = 1; y < LEN - 1; y = y + 2)
    {
        // 先判定头部第一个格子与外界的边是否自由边
        if (getFreeBoxBool(1, y) && (map[0][y] == HENG)) // 第一个为交格而且与外界交互的边为空边
        {
            if (getSaveAngleEdgeBool(1, y, st))
            {
                Moves[MoveNum] = {0, y}; // 保存坐标
                MoveNum++;               // 总自由边数目自增1
            }
            // else
            //   cout<<"("<<0<<","<<y<<")"<<endl;
            st[0][y] = true;
        }
        // 循环判定中间的几个格子
        for (int x = 1; x < LEN - 3; x = x + 2) // x轴
        {
            if (getFreeBoxBool(x, y) && getFreeBoxBool(x + 2, y) && (map[x + 1][y] == HENG))
            {
                Moves[MoveNum] = {x + 1, y}; // 保存坐标
                MoveNum++;                   // 总自由边数目自增1
            }
            st[x + 1][y] = true;
        }
        // 判断末尾的格子
        if (getFreeBoxBool(LEN - 2, y) && (map[LEN - 1][y] == HENG)) // 最后一个为交格且与外界交互的边为空边
        {
            if (getSaveAngleEdgeBool(LEN - 2, y, st))
            {
                Moves[MoveNum] = {LEN - 1, y}; // 保存坐标
                MoveNum++;                     // 总自由边数目自增1
            }
            // else
            //    cout<<"("<<LEN-1<<","<<y<<")"<<endl;
            st[LEN - 1][y] = true;
        }

        // XY替换，再进行一次判定

        // 先判定头部第一个格子与外界的边是否自由边
        if (getFreeBoxBool(y, 1) && (map[y][0] == SHU)) // 第一个为交格而且与外界交互的边为空边
        {
            if (getSaveAngleEdgeBool(y, 1, st))
            {
                Moves[MoveNum] = {y, 0}; // 保存坐标
                MoveNum++;               // 总自由边数目自增1
            }
            // else
            //   cout<<"("<<y<<","<<0<<")"<<endl;
            st[y][0] = true;
        }
        // 循环判定中间的几个格子
        for (int x = 1; x < LEN - 3; x = x + 2) // x轴
        {
            if (getFreeBoxBool(y, x) && getFreeBoxBool(y, x + 2) && (map[y][x + 1] == SHU))
            {
                Moves[MoveNum] = {y, x + 1}; // 保存坐标
                MoveNum++;                   // 总自由边数目自增1
            }
            st[y][x + 1] = true;
        }
        // 判断末尾的格子
        if (getFreeBoxBool(y, LEN - 2) && (map[y][LEN - 1] == SHU)) // 最后一个为交格且与外界交互的边为空边
        {
            if (getSaveAngleEdgeBool(y, LEN - 2, st))
            {
                Moves[MoveNum] = {y, LEN - 1}; // 保存坐标
                MoveNum++;                     // 总自由边数目自增1
            }
            // else
            //   cout<<"("<<y<<","<<LEN-1<<")"<<endl;
            st[y][LEN - 1] = true;
        }
    }
    return MoveNum; // 返回自由边的总数
}