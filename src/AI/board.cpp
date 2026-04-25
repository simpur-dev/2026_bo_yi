#include "board.h"

/**
 * @brief 默认构造函数，分为DOT，BOX，HENG，SHU。
 * @details 同为偶是点，同为奇是格子，先偶后奇是横线，先奇后偶是竖线
 */
Board::Board()
{
    for (int i = 0; i < 11; ++i)
    {
        for (int j = 0; j < 11; ++j)
        {
            if (i % 2 == 0 && j % 2 == 0)
            {
                map[i][j] = DOT;
            }
            else if (i % 2 == 0 && j % 2 == 1)
            {
                map[i][j] = HENG;
            }
            else if (i % 2 == 1 && j % 2 == 1)
            {
                map[i][j] = BOX;
            }
            else
            {
                map[i][j] = SHU;
            }
        }
    }
    blackBox = whiteBox = 0;
}

Board::Board(int Array[LEN][LEN])
{
    for (int i = 0; i < LEN; i++)
    {
        for (int j = 0; j < LEN; j++)
        {
            map[i][j] = Array[i][j];
        }
    }
    blackBox = whiteBox = 0;
    for (int i = 1; i < 11; i += 2)
    {
        for (int j = 1; j < 11; j += 2)
        {
            if (map[i][j] == BLACK)
            {
                blackBox++;
            }
            else if (map[i][j] == WHITE)
            {
                whiteBox++;
            }
        }
    }
}

/**
 * @brief 获得一个格子的自由度，自由度即格子周围未被占领的边的数量
 *
 * @param x 横坐标
 * @param y 纵坐标
 * @return int 自由度
 */
int Board::getFreedom(int x, int y)
{
    int cnt = 4;
    if (map[x - 1][y] == OCCLINE)
        --cnt;
    if (map[x + 1][y] == OCCLINE)
        --cnt;
    if (map[x][y - 1] == OCCLINE)
        --cnt;
    if (map[x][y + 1] == OCCLINE)
        --cnt;
    return cnt;
}

/**
 * @brief 通过LOC返回自由度
 *
 * @param l 坐标
 * @return int 自由度
 */
int Board::getFreedom(LOC l)
{
    return getFreedom(l.first, l.second);
}

/**
 * @brief 占线
 * @details
 * 默认传进来的线都是没有被占过的，如果传进来的线是横线，要考虑它上下两个格子自由度是否为3；如果已经是3，那么再占一条线，这个格子就属于传进来下棋方的。同时要考虑边界情况，比如横线在最上面时，只需考虑线下面的格子，同理，如果是竖线，就要考虑左右两个格子
 *
 * @param player 代表下棋方
 * @param l 代表要占的线的坐标
 * @return int 返回的是这条线产生的被占了的格子的数目
 */
int Board::move(int player, LOC l)
{
    int res = 0;
    // 如果传进来的坐标不是边，输出
    if (!isFreeLine(l))
    {
        // cerr << "Board::move(LOC l): " << l.first << " " << l.second << "\n";
        return 0;
    }
    if (map[l.first][l.second] == HENG)
    {
        if (l.first == 0)
        {
            if (getFreedom(1, l.second) == 1)
            {
                map[1][l.second] = player;
                ++res;
            }
        }
        else if (l.first == 10)
        {
            if (getFreedom(9, l.second) == 1)
            {
                map[9][l.second] = player;
                ++res;
            }
        }
        else
        {
            if (getFreedom(l.first - 1, l.second) == 1)
            {
                map[l.first - 1][l.second] = player;
                ++res;
            }
            if (getFreedom(l.first + 1, l.second) == 1)
            {
                map[l.first + 1][l.second] = player;
                ++res;
            }
        }
    }
    else
    {
        if (l.second == 0)
        {
            if (getFreedom(l.first, 1) == 1)
            {
                map[l.first][1] = player;
                ++res;
            }
        }
        else if (l.second == 10)
        {
            if (getFreedom(l.first, 9) == 1)
            {
                map[l.first][9] = player;
                ++res;
            }
        }
        else
        {
            if (getFreedom(l.first, l.second - 1) == 1)
            {
                map[l.first][l.second - 1] = player;
                ++res;
            }
            if (getFreedom(l.first, l.second + 1) == 1)
            {
                map[l.first][l.second + 1] = player;
                ++res;
            }
        }
    }
    map[l.first][l.second] = OCCLINE;
    if (player == BLACK)
        blackBox += res;
    else
        whiteBox += res;
    return res;
}

/**
 * @brief 吃C型格，
 *
 * @param player 代表此时的下棋方
 * @return LOC 返回一个C型格的坐标，找不到返回{-1，-1}
 */
LOC Board::eatCBox(int player)
{
    for (int i = 1; i < 11; i += 2)
    {
        for (int j = 1; j < 11; j += 2)
        {
            if (getFreedom(i, j) == 1)
            {
                if (map[i - 1][j] != OCCLINE)
                {
                    move(player, {i - 1, j});
                    return {i - 1, j};
                }
                else if (map[i + 1][j] != OCCLINE)
                {
                    move(player, {i + 1, j});
                    return {i + 1, j};
                }
                else if (map[i][j + 1] != OCCLINE)
                {
                    move(player, {i, j + 1});
                    return {i, j + 1};
                }
                else
                {
                    move(player, {i, j - 1});
                    return {i, j - 1};
                }
            }
        }
    }
    // 找不到默认返回{-1，-1}
    return {-1, -1};
}

bool Board::isFreeLine(LOC l) const
{
    if (map[l.first][l.second] != SHU && map[l.first][l.second] != HENG)
        return false;
    else
        return true;
}

bool Board::isFreeLine(int i, int j) const
{
    if (map[i][j] != SHU && map[i][j] != HENG)
        return false;
    else
        return true;
}

/**
 * @brief 判断一个坐标是否会造成C型格
 *
 * @return ture表示会造成C型格
 */
bool Board::ifMakeCBox(LOC l)
{
    if (map[l.first][l.second] == HENG)
    {
        if (l.first == 0)
        {
            if (getFreedom(l.first+1, l.second ) == 2)
                return true;
            else
                return false;
        }
        else if (l.first == 10)
        {
            if (getFreedom(l.first - 1, l.second) == 2)
                return true;
            else
                return false;
        }
        else
        {
            if (getFreedom(l.first-1, l.second) == 2 || getFreedom(l.first+1, l.second) == 2)
                return true;
            else
                return false;
        }
    }
    else
    {
        if (l.second == 0)
        {
            if (getFreedom(l.first , l.second+1) == 2)
                return true;
            else
                return false;
        }
        else if (l.second == 10)
        {
            if (getFreedom(l.first , l.second-1) == 2)
                return true;
            else
                return false;
        }
        else
        {
            if (getFreedom(l.first , l.second-1) == 2 || getFreedom(l.first , l.second+1) == 2)
                return true;
            else
                return false;
        }
    }
}

/**
 * @brief 判断是否赢得格子
 *
 * @param l 需要判断的边的坐标
 * @return 如果传进来的l占据了格子，返回true，否则返回false
 */
bool Board::ifEarnBox(LOC l)
{
    if (map[l.first][l.second] == HENG)
    {
        if (l.first == 0)
        {
            if (getFreedom(1, l.second) == 1)
            {
                return true;
            }
        }
        else if (l.first == 10)
        {
            if (getFreedom(9, l.second) == 1)
            {
                return true;
            }
        }
        else
        {
            if (getFreedom(l.first - 1, l.second) == 1)
            {
                return true;
            }
            if (getFreedom(l.first + 1, l.second) == 1)
            {
                return true;
            }
        }
    }
    else
    {
        if (l.second == 0)
        {
            if (getFreedom(l.first, 1) == 1)
            {
                return true;
            }
        }
        else if (l.second == 10)
        {
            if (getFreedom(l.first, 9) == 1)
            {
                return true;
            }
        }
        else
        {
            if (getFreedom(l.first, l.second - 1) == 1)
            {
                return true;
            }
            if (getFreedom(l.first, l.second + 1) == 1)
            {
                return true;
            }
        }
    }
    return false;
}

/**
 * @brief 判断棋局是否结束
 *
 * @return 如果游戏已经结束，返回true，否则返回false
 */
bool Board::ifEnd() const
{
    if (blackBox + whiteBox == 25)
    {
        //        if (blackBox > 12 || whiteBox > 12)
        return true;
    }
    else
        return false;
}

/**
 * @brief 判断赢家
 *
 * @return 如果黑方胜利，返回BLACK，白方胜利返回WHITE，目前无胜方返回EMPTY
 */
int Board::getWinner() const
{
    if (blackBox + whiteBox != 25)
        return EMPTY;
    if (blackBox > whiteBox)
        return BLACK;
    else
        return WHITE;
}

bool Board::getCTypeBoxLimit(int Player, vector<LOC> &pace)
{
    // 仅用在前期，用于在搜索的时候占领所有C型格
    for (int y = 1; y < LEN - 1; y = y + 2)
    {
        for (int x = 1; x < LEN - 1; x = x + 2) // x轴
        {
            if (getCTypeBoxBool(x, y))
            {
                if ((map[x + 1][y] == HENG || map[x + 1][y] == SHU) && !getCTypeBoxBool(x + 2, y))
                {
                    move(Player, {x + 1, y});
                    pace.emplace_back(x + 1, y); //**记录步伐
                    return true;                 // 占据之后就返回真
                }
                else if ((map[x - 1][y] == HENG || map[x - 1][y] == SHU) && !getCTypeBoxBool(x - 2, y))
                {
                    move(Player, {x - 1, y});
                    pace.emplace_back(x - 1, y); //**记录步伐
                    return true;                 // 占据之后就返回真
                }
                else if ((map[x][y + 1] == HENG || map[x][y + 1] == SHU) && !getCTypeBoxBool(x, y + 2))
                {
                    move(Player, {x, y + 1});
                    pace.emplace_back(x, y + 1); //**记录步伐
                    return true;                 // 占据之后就返回真
                }
                else if ((map[x][y - 1] == SHU || map[x][y - 1] == SHU) && !getCTypeBoxBool(x, y - 2))
                {
                    move(Player, {x, y - 1});
                    pace.emplace_back(x, y - 1); //*记录步伐
                    return true;                 // 占据之后就返回真
                }
            }
        }
    }
    return false; // 返回假
}

void boardCopy(int Source[LEN][LEN], int Target[LEN][LEN])
{
    for (int y = 0; y < LEN; y++)
    {
        for (int x = 0; x < LEN; x++)
        {
            Target[x][y] = Source[x][y];
        }
    }
}

void Board::setBoard(int Array[LEN][LEN])
{
    for (int i = 0; i < LEN; i++)
    {
        for (int j = 0; j < LEN; j++)
        {
            map[i][j] = Array[i][j];
        }
    }
    blackBox = whiteBox = 0;
    for (int i = 1; i < 11; i += 2)
    {
        for (int j = 1; j < 11; j += 2)
        {
            if (map[i][j] == BLACK)
            {
                blackBox++;
            }
            else if (map[i][j] == WHITE)
            {
                whiteBox++;
            }
        }
    }
}

int Board::getFilterMoveNum()
{
    int MoveNum = 0;
    //横
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
                        MoveNum++; // 总数目自增
                    }
                }
                else if (x == LEN - 1)
                {
                    if (!getLongCTypeBoxBool(x - 1, y)) // 如果左边的那个格子没问题的话，这个招法也没问题
                    {
                        MoveNum++; // 总数目自增
                    }
                }
                else
                {
                    if (!getLongCTypeBoxBool(x + 1, y) &&
                        !getLongCTypeBoxBool(x - 1, y)) // 如果左右两边的格子都没问题的话，这个招法也没问题
                    {
                        MoveNum++; // 总数目自增
                    }
                }
                setBoard(BoardSave); // 还原
            }
        }
    }
    //竖
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
                        MoveNum++; // 总数目自增
                    }
                }
                else if (y == LEN - 1)
                {
                    if (!getLongCTypeBoxBool(x, y - 1)) // 如果上面的那个格子没问题的话，这个招法也没问题
                    {
                        MoveNum++; // 总数目自增
                    }
                }
                else
                {
                    if (!getLongCTypeBoxBool(x, y + 1) &&
                        !getLongCTypeBoxBool(x, y - 1)) // 如果上下的格子都没问题的话，这个招法也没问题
                    {
                        MoveNum++; // 总数目自增
                    }
                }
                setBoard(BoardSave); // 还原
            }
        }
    }

    return MoveNum;
}

bool Board::getCTypeBoxBool(int bx, int by)
{
    //#注意的是，输入的X与Y是Box实际地址
    if (bx >= 1 && bx <= LEN - 2 && by >= 1 && by <= LEN - 2 && isOdd(bx) && isOdd(by)) // Box位置编号必须正确
    {
        if (getFreedom(bx, by) == DEADBOX)
            return true;
        else
            return false;
    }
    return false;
}

void Board::eatAllCTypeBoxes(int Player, vector<LOC> &pace)
{
    LOC flag = {-1, -1};
    for (;;) // 直到无法占据CTypeBox了就结束
    {
        LOC tmp = eatCBox(Player);
        if (tmp == flag)
            break;
        else
            pace.emplace_back(tmp); //**记录步伐
    }
}
void Board::eatAllCTypeBoxes(int Player)
{
    LOC flag = {-1, -1};
    for (;;) // 直到无法占据CTypeBox了就结束
    {
        LOC tmp = eatCBox(Player);
        if (tmp == flag)
            break;
    }
}

bool Board::getLongCTypeBoxExist()
{
    for (int by = 1; by < LEN - 1; by = by + 2)
    {
        for (int bx = 1; bx < LEN - 1; bx = bx + 2)
        {
            if (getFreedom(bx, by) == DEADBOX) // 如果存在自由度为1的格子
            {
                for (auto n : Dir)
                {
                    int ex = bx + n[0];
                    int ey = by + n[1];
                    int nbx = bx + n[0] + n[0];
                    int nby = by + n[1] + n[1];
                    if ((map[ex][ey] == HENG || map[ex][ey] == SHU) && nbx >= 1 && nbx <= LEN - 2 && nby >= 1 &&
                        nby <= LEN - 2)
                    {
                        if (getFreedom(nbx, nby) == CHAINBOX)
                            return true;
                    }
                }
            }
        }
    }
    return false;
}

int Board::getFreeEdgeNum()
{
    int EdgeNum = 0;
    // 得到所有的自由边
    for (int y = 1; y < LEN - 1; y = y + 2)
    {
        // 先判定头部第一个格子与外界的边是否自由边
        if (getFreeBoxBool(1, y) && map[0][y] == HENG) // 第一个为交格而且与外界交互的边为空边
        {
            EdgeNum++; // 总自由边数目自增1
        }
        // 循环判定中间的几个格子
        for (int x = 1; x < LEN - 3; x = x + 2) // x轴
        {
            if (getFreeBoxBool(x, y) && getFreeBoxBool(x + 2, y) && map[x + 1][y] == HENG)
            {
                EdgeNum++; // 总自由边数目自增1
            }
        }
        // 判断末尾的格子
        if (getFreeBoxBool(LEN - 2, y) && map[LEN - 1][y] == HENG) // 最后一个为交格且与外界交互的边为空边
        {
            EdgeNum++; // 总自由边数目自增1
        }

        // XY替换，再进行一次判定

        // 先判定头部第一个格子与外界的边是否自由边
        if (getFreeBoxBool(y, 1) && map[y][0] == SHU) // 第一个为交格而且与外界交互的边为空边
        {
            EdgeNum++; // 总自由边数目自增1
        }
        // 循环判定中间的几个格子
        for (int x = 1; x < LEN - 3; x = x + 2) // x轴
        {
            if (getFreeBoxBool(y, x) && getFreeBoxBool(y, x + 2) && map[y][x + 1] == SHU)
            {
                EdgeNum++; // 总自由边数目自增1
            }
        }
        // 判断末尾的格子
        if (getFreeBoxBool(y, LEN - 2) && map[y][LEN - 1] == SHU) // 最后一个为交格且与外界交互的边为空边
        {
            EdgeNum++; // 总自由边数目自增1
        }
    }
    return EdgeNum; // 返回自由边的总数
}

bool Board::getFreeBoxBool(int bx, int by)
{
    // 注意的是，输入的X与Y是Box实际地址
    if (1 <= bx && LEN - 2 >= bx && 1 <= by && LEN - 2 >= by && isOdd(bx) && isOdd(by)) // Box位置编号必须正确
    {
        if (getFreedom(bx, by) >= 3)
            return true;
        else
            return false;
    }
    // else
    // {
    //     // cout << "Wrong Number In <GetFreeBoxBool> Function";
    //     // system("pause");
    // }
    return false;
}
void Board::unmove(LOC l)
{
    if (l.first % 2 == 0 && l.second % 2 == 1)
    { // 横边
        if (l.first == 0)
        {
            switch (map[l.first + 1][l.second])
            {
            case BLACK:
                blackBox--;
                map[l.first + 1][l.second] = EMPTY;
                break;
            case WHITE:
                whiteBox--;
                map[l.first + 1][l.second] = EMPTY;
                break;
            default:
                break;
            }
        }
        else if (l.first == 10)
        {
            switch (map[l.first - 1][l.second])
            {
            case BLACK:
                blackBox--;
                map[l.first - 1][l.second] = EMPTY;
                break;
            case WHITE:
                whiteBox--;
                map[l.first - 1][l.second] = EMPTY;
                break;
            default:
                break;
            }
        }
        else
        {
            switch (map[l.first + 1][l.second])
            {
            case BLACK:
                blackBox--;
                map[l.first + 1][l.second] = EMPTY;
                break;
            case WHITE:
                whiteBox--;
                map[l.first + 1][l.second] = EMPTY;
                break;
            default:
                break;
            }
            switch (map[l.first - 1][l.second])
            {
            case BLACK:
                blackBox--;
                map[l.first - 1][l.second] = EMPTY;
                break;
            case WHITE:
                whiteBox--;
                map[l.first - 1][l.second] = EMPTY;
                break;
            default:
                break;
            }
        }
        map[l.first][l.second] = HENG;
    }
    else if (l.first % 2 == 1 && l.second % 2 == 0)
    { // 竖边
        if (l.second == 0)
        {
            switch (map[l.first][l.second + 1])
            {
            case BLACK:
                blackBox--;
                map[l.first][l.second + 1] = EMPTY;
                break;
            case WHITE:
                whiteBox--;
                map[l.first][l.second + 1] = EMPTY;
                break;
            default:
                break;
            }
        }
        else if (l.second == 10)
        {
            switch (map[l.first][l.second - 1])
            {
            case BLACK:
                blackBox--;
                map[l.first][l.second - 1] = EMPTY;
                break;
            case WHITE:
                whiteBox--;
                map[l.first][l.second - 1] = EMPTY;
                break;
            default:
                break;
            }
        }
        else
        {
            switch (map[l.first][l.second + 1])
            {
            case BLACK:
                blackBox--;
                map[l.first][l.second + 1] = EMPTY;
                break;
            case WHITE:
                whiteBox--;
                map[l.first][l.second + 1] = EMPTY;
                break;
            default:
                break;
            }
            switch (map[l.first][l.second - 1])
            {
            case BLACK:
                blackBox--;
                map[l.first][l.second - 1] = EMPTY;
                break;
            case WHITE:
                whiteBox--;
                map[l.first][l.second - 1] = EMPTY;
                break;
            default:
                break;
            }
        }
        map[l.first][l.second] = SHU;
    }
    else
    {
        // cout << "error: Board::unmove()" << endl;
    }
}
bool Board::ifHaveSafeEdge()
{
    for (int i = 1; i < 11; i += 2)
    {
        for (int j = 0; j < 11; j += 2)
        {
            if (map[i][j] != OCCLINE && !ifMakeCBox(LOC{i, j}))
            {
                return true;
            }
        }
    }
    for (int i = 0; i < 11; i += 2)
    {
        for (int j = 1; j < 11; j += 2)
        {
            if (map[i][j] != OCCLINE && !ifMakeCBox(LOC{i, j}))
            {
                return true;
            }
        }
    }
    return false;
}
LOC Board::getPublicSide(LOC a, LOC b) // 传入新旧坐标
{
    LOC res = {-1, -1};
    LOC A, B;
    A.first = a.first * 2 - 1, A.second = a.second * 2 - 1;
    B.first = b.first * 2 - 1, B.second = b.second * 2 - 1;
    if (A.first == B.first && A.second == (B.second - 2))
        res = {B.first, B.second - 1}; // a在b正上方
    if (A.first == B.first && A.second == (B.second + 2))
        res = {B.first, B.second + 1}; // a在b正下方
    if (A.second == B.second && A.first == (B.first - 2))
        res = {B.first - 1, B.second}; // a在b正左方
    if (A.second == B.second && A.first == (B.first + 2))
        res = {B.first + 1, B.second}; // a在b正右方
    return res;
}

int Board::getPlayerBoxes(int player) const
{
    switch (player)
    {
    case BLACK:
        return blackBox;
        break;
    case WHITE:
        return whiteBox;
        break;
    default:
        return 0;
        break;
    }
    // int boxes = 0;
    // if (player == BLACK)
    // {
    //     for (int i = 0; i < LEN; i++)
    //     {
    //         for (int j = 0; j < LEN; j++)
    //         {
    //             if (map[i][j] == BLACK)
    //                 boxes++;
    //         }
    //     }
    // }
    // else if (player == WHITE)
    // {
    //     for (int i = 0; i < LEN; i++)
    //     {
    //         for (int j = 0; j < LEN; j++)
    //         {
    //             if (map[i][j] == WHITE)
    //                 boxes++;
    //         }
    //     }
    // }
    // return boxes;
}

bool isOdd(int num)
{
    // 判断一个数字是否是奇数
    if (num % 2 != 0)
        return true;
    return false;
}
bool isEven(int num)
{
    // 判断一个数字是否是偶数
    if (num % 2 == 0)
        return true;
    return false;
}

bool Board::getLongCTypeBoxBool(int bx, int by) // 判断一个格子是否是一条长死格的起点
{
    if (getFreedom(bx, by) == DEADBOX) // 首先这个格子必须本身是一个C型格
    {
        for (auto n : Dir)
        {
            int ex = bx + n[0];
            int ey = by + n[1];
            int nbx = bx + n[0] + n[0]; // 下一个格子的实际地址
            int nby = by + n[1] + n[1]; // 下一个格子的实际地址
            if ((map[ex][ey] == HENG || map[ex][ey] == SHU) && nbx >= 1 && nbx <= LEN - 2 && nby >= 1 && nby <= LEN - 2)
            {
                if (getFreedom(nbx, nby) == CHAINBOX)
                    return true;
            }
        }
    }
    return false;
}
LOC Board::getDoubleCrossLoc(int Player)
{
    // 得到可以制造双交的那个边的坐标
    for (int by = 1; by < LEN - 1; by = by + 2)
    {
        for (int bx = 1; bx < LEN - 1; bx = bx + 2)
        {
            if (getLongCTypeBoxBool(bx, by))
            {
                // 现在的bx和by就是实际的格子
                for (auto n : Dir)
                {
                    int ex = bx + n[0];
                    int ey = by + n[1];
                    int nbx = bx + n[0] + n[0];
                    int nby = by + n[1] + n[1];
                    if ((map[ex][ey] == SHU | map[ex][ey] == HENG) && nbx >= 1 && nbx <= LEN - 2 && nby >= 1 &&
                        nby <= LEN - 2)
                    {
                        // 现在ex,ey是其公共边，nbx,nby是doublecross的末端CHAINBOX
                        for (auto dir : Dir)
                        {
                            int nex = nbx + dir[0];
                            int ney = nby + dir[1];
                            if ((map[nex][ney] == HENG || map[nex][ney] == SHU) &&
                                (nex != ex || ney != ey)) // 空边而且不是中间的公共边(ex,ey)
                            {
                                LOC k;
                                k = {nex, ney};
                                return k;
                            }
                        }
                    }
                }
            }
        }
    }
    LOC k;
    k = {0, 0};
    return k;
}