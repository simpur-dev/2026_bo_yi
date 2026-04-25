#include "AI/Node.h"
#include "AI/UCT.h"
#include "AI/board.h"
#include "AI/define.h"
#include "CJSON/datarecorder.h"
#include <iostream>
#include <string>
#include <vector>

using std::cin;
using std::cout;
using std::endl;
using std::string;

string name = "SingleThread"; //程序名字
bool recordBoard = 0;         //是否打印棋盘

Board *brd;
Step steps[60];
int top = -1;

string BlackName;
string WhiteName;

int temp = 0;
// 其中"name?","new","move","end","quit","error"为平台向引擎传递命令；
// "name","move"为引擎向平台传递命令字
string change(LOC l);
LOC change(string l);

int main()
{
    string message;
    int ai;
    while (true)
    {
        cin >> message;
        if (message == "move")
        {
            int n;
            cin >> n >> message;
            for (int i = 0; i < message.size(); i += 3)
            {
                string t;
                t += message[i];
                t += message[i + 1];
                t += message[i + 2];
                // 占线
                brd->move(-ai, change(t));
                steps[++top] = {-ai == BLACK ? BLACK : WHITE, change(t)};
            }
            vector<LOC> moves;
            gameTurnMove(*brd, ai, &temp, moves);
            string res;
            for (auto const &i : moves)
            {
                res += change(i);
                steps[++top] = {ai == BLACK ? BLACK : WHITE, i};
            }
            cout << "move " << res.size() / 3 << " " << res << endl;
        }
        else if (message == "name?")
        {
            cout << "name " << name << endl;
        }
        else if (message == "new")
        {
            time_t now = time(NULL);
            BlackName = std::to_string(localtime(&now)->tm_min);
            cin >> message;
            brd = new Board;
            top = -1;
            if (message == "black")
            {
                ai = BLACK;
                vector<LOC> moves;
                gameTurnMove(*brd, ai, &temp, moves);
                string res;
                for (auto const &i : moves)
                {
                    res += change(i);
                    steps[++top] = {BLACK, i};
                }
                cout << "move " << res.size() / 3 << " " << res << endl;
            }
            else
            {
                ai = WHITE;
            }
        }
        else if (message == "error")
        {
            cout << "error! check it!" << endl;
        }
        else if (message == "end")
        {
            time_t now = time(NULL);
            WhiteName = std::to_string(localtime(&now)->tm_min);
            delete brd;
            data_of_game ginfo = data_of_game(brd->blackBox, brd->whiteBox, BlackName.c_str(), WhiteName.c_str());
            ginfo.recordstep(steps);
            ginfo.endrecord();
            ginfo.printdata();
            ginfo.deleterecord();
        }
        else if (message == "quit")
        {
            return 0;
        }
    }
}

string change(LOC l)
{
    string s;
    if (l.first % 2 == 0 && l.second % 2 == 1)
    {
        s += 'A';
        s += 'A' + l.first / 2;
        s += 'A' + l.second / 2;
    }
    else
    {
        s += 'B';
        s += 'A' + l.second / 2;
        s += 'A' + l.first / 2;
    }
    return s;
}

/**
 * @brief 将字符串转换成坐标
 * @param l 对战平台传进的坐标值
 * @return 坐标值
 */
LOC change(string l)
{
    LOC res;
    if (l[0] == 'A')
    {
        res.first = (l[1] - 'A') * 2;
        res.second = (l[2] - 'A') * 2 + 1;
    }
    else
    {
        res.second = (l[1] - 'A') * 2;
        res.first = (l[2] - 'A') * 2 + 1;
    }
    return res;
}