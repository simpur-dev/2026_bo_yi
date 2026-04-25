#include "../AI/define.h"
#include "cJSON.h"
#include <cstring>
#include <fstream>
#include <iostream>

class data_of_game
{

  public:
    cJSON *root;  // json根节点
    cJSON *piece; // json落子信息数组节点
    cJSON *objarr;
    const char *winner; //胜者
    const char *win;
    const char *pf; //先手名字
    const char *ps; //后手名字

    //创建json记录器，需要输入先后手得分，时间，先后手玩家名字，胜者名字
    data_of_game(int a, int b, const char *p1, const char *p2);

    // 录入步骤信息传入vector<step>
    void recordstep(Step v[60]);

    //当所有步骤都被录入就调用此函数结束录入
    void endrecord();

    //打印生成的结果输入三个参数，先手名字，后手名字，获胜者名字用于创建txt文件名
    void printdata();

    //释放json节点的空间
    void deleterecord();
};
