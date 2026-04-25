#include "datarecorder.h"
#include <fstream>
//#include <fstream.hpp>
using std::fstream;
using std::ios;

data_of_game::data_of_game(int a, int b, const char *p1, const char *p2)
{
    root = cJSON_CreateObject(); // json根节点
    piece = cJSON_CreateArray(); // json落子信息数组节点
    cJSON_AddStringToObject(root, "R", p1);
    this->pf = p1;
    cJSON_AddStringToObject(root, "B", p2);
    this->ps = p2;
    if (a + b == 25)
    {
        if (a > b)
        {
            winner = pf;
            win = "R";
        }
        else
        {
            winner = ps;
            win = "B";
        }
    }
    else
    {
        winner = "未分出";
        win = "U";
    }
    cJSON_AddStringToObject(root, "winner", win);
    cJSON_AddNumberToObject(root, "RScore", a);
    cJSON_AddNumberToObject(root, "BScore", b);
  //  time_t now = time(nullptr);
 //   tm *date = localtime(&now);
    cJSON_AddStringToObject(root, "Date", "2022-08-06");
    cJSON_AddStringToObject(root, "Event", "2022 CCGC");
}

void data_of_game::recordstep(Step v[60]) // x行y列
{
    for (int i = 0; i < 60; i++)
    {
        if (v[i].player == EMPTY)
            continue;
        std::string tem;
        tem += v[i].player == -1 ? "b(" : "r(";
        tem += (v[i].action.second / 2 + 'a');
        tem += (6 - ((v[i].action.first + 1) / 2) + '0');
        tem += (v[i].action.first % 2 == 0 ? ",h)" : ",v)");
        cJSON *node = cJSON_CreateObject();
        cJSON_AddStringToObject(node, "piece", tem.c_str());
        cJSON_AddItemToArray(piece, node);
    }
}

void data_of_game::endrecord()
{
    cJSON_AddItemToObject(root, "game", piece);
}

//打印生成的结果
void data_of_game::printdata()
{
    //（例如  A1：A1先手，A2后手，A1胜）
    char *json_data = cJSON_PrintUnformatted(root);
//    fstream out;
    char buf[64];
    sprintf(buf, "A6：%s先手，%s后手，%s胜.txt", pf, ps, winner);
//    out.open(buf, ios::out);
//    out << json_data;
//     cout << filename;
//    out.close();
}

void data_of_game::deleterecord() //释放json节点的空间
{
    cJSON_Delete(root);
}
