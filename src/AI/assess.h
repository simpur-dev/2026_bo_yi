#include "board.h"
#include "define.h"

#pragma once

// BoxType类是一个用于鉴定格子类型的类
class BoxType
{
  public:
    BoxType();
    int Type;              //[0]已被占领 [1]死格 [2]链格 [3]自由格
    int BelongingChainNum; //所属链类型的编号，会从1开始自动编号。
};

// ChainInfo是一个包含了一条链的基本信息的类
class ChainInfo
{
  public:
    ChainInfo();
    ChainType Type;
    int ChainBoxNum;
    LOC StartLoc;
    LOC EndLoc;
    bool ConditionOfPreCircle; //预备环的先决条件
};

class BoxBoard : public Board
{
  public:
    BoxBoard(Board NewB);
    BoxBoard(int map[LEN][LEN]);
    BoxType Boxes[BOXLEN + 1][BOXLEN + 1]; // 6*6。后期棋盘的基本类型(*占领/未被占领*)，编号从1开始。
    ChainInfo Chains[25];                  //链的数量必定不会超过全部格子数

    //注册链
    //说明：链是动态注册的。当一条链被归类到另一条链时候，他本身会被重新定义为"NotDefine"，而每次需要注册链的时候会从0开始查找到一条空链。
    LOC findNextBox(LOC FoundBox, LOC LastBox); //查找某个链格
    int getFirstEmptyChainNum();                //功能：获得一个空的链编号
    int getBoxType(int bx, int by); //用于得到某个格子的类型  不在坐标范围内的全部返回自由格类型[坐标限制为1到BOXLEN]。
    void inheritChain(int InheritorRegNum, int AncesterRegNum); //一条链吞并另一条链
    void registerChain(LOC FreeBoxLoc, LOC NextLoc); //从一个场内的自由格出发，给所有的派生链做一个定义。
    void registerCircle(LOC StartLoc, LOC NextLoc); //从一个格出发，确定是否为一个环。
    void searchingFromBox(LOC BoxLoc); //从一个格出发，注册他的所有派生链,ChianPlus应仅在没有短链时启用。
    void searchingCircle(LOC BoxLoc);     //从一个格出发，注册他的所有派生环。
    void defineBoxesType();               //将所有格子的基本信息填入，包括属主和格子类型
    void defineAllChains(bool ChainPlus); //定义所有的链
    void resetChainsInfo();               //重定义链与格的的信息
    void defineDeadChain();               //定义死环或者死链
    void searchingDeadChain(LOC BoxLoc);  //
    void registerDeadChain(LOC FreeBoxLoc, LOC FirstLoc); //注册死环或者死链

    //鉴别链
    bool getDeadChainExist();  //*判断死链是否存在
    bool getDeadCircleExist(); //*判读死环是否存在

    //占据链
    bool captualShortestChain(int LatterPlayer); //*吞并一条当前最短的长链或者环
    LOC getOpenShortestChainLoc();               //*获得打开最短长链的坐标
    LOC openPolicy();                            //*打开策略
    LOC getOpenShortestCircleLoc();              //*获得打开最短环的坐标
    LOC getOpen3ChainLoc();                      //*获得打开3链的坐标

    //最终判定局面
    static bool rationalState(LOC BoxNum);
    //*判断双方是否为理性状态。理性状态：当前局面中，预测控制者一直保持控制得到的格子数，大于让给对方的格子数
    LOC getEarlyRationalBoxNum(); //*用在UCT前的预处理中
    LOC getRationalStateBoxNum(); //*用在后期决策中
    //*获得当前局面下双方以理性状态可以获得的格子数量
    //*x为控制者在余下局面一直保持控制权得到的格子数,y为打开者获得的格子数（在余下局面中控制者让给打开者的格子数）
    int getBoardWinner(int LatterPlayer);             //*得到或预测本局面下的胜利者，参数为后手方。
    int getBoardWinner(int LatterPlayer, int FaOwner,int &score); //*得到或预测本局面下的胜利者，参数为后手方。

    //等价裁边
    int getFilterMoves(LOC Moves[60]);                          //*得到所有的过滤可行边(不产生死链)
    int getFreeMoves(LOC Moves[60]);                            //*得到所有的自由边，同时存下坐标
    bool getSaveChainEdgeBool(int x, int y, bool st[LEN][LEN]); //一格短链的裁边
    bool getSaveAngleEdgeBool(int x, int y, bool st[LEN][LEN]); //边角格子的裁边
};
