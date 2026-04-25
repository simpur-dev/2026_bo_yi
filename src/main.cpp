#include "../src/CJSON/cJSON.h"
#include "../src/CJSON/datarecorder.h"
#include "AI/UCT.h"
#include "AI/board.h"
#include "AI/define.h"
#include "element/Time.h"
#include <SFML/Graphics.hpp>
#include <fstream>
#include <future>
#include <thread>

std::string BlackName = "先手";
std::string WhiteName = "后手";

sf::RenderWindow mainWindow;
sf::Font font;
sf::Texture texture;
sf::Sprite sprite;

Board *gameBoard;
int nowPlayer = BLACK;
LOC nowMove = {-1, -1};

Time black_time;
Time white_time;

bool black_ai = false;
bool white_ai = false;
int game_begin = 0;

Step steps[60];
int top = -1;
vector<LOC> ai_steps;
////////显示左侧棋盘函数//////////
void showGameBoard();
void showVisualLine();
void showLine();
void showBox();
void showDots();
///////显示右侧信息和按钮///////
sf::CircleShape now_player(10);
sf::RectangleShape first_button(sf::Vector2f(100, 80));
sf::RectangleShape second_button(sf::Vector2f(100, 80));
//////游戏开始////////
sf::RectangleShape GameButton(sf::Vector2f(300.f, 100.f));
///////undo和redo/////
sf::RectangleShape undo_button(sf::Vector2f(150.f, 100.f));
sf::RectangleShape redo_button(sf::Vector2f(150.f, 100.f));
///////打印棋盘////////
sf::RectangleShape print_button(sf::Vector2f(300, 100));
///////加载棋盘////////
sf::RectangleShape load_button(sf::Vector2f(300, 100));
void initSidebar();
void showSidebar();
void showNowPlayer();
void showInformation();
void showGameBegin();
void showUndoAndRedo();
void showPrintBoard();
void showSetPlayer();
void showLoadBoard();
////////按键处理//////////
void handleButtons(int x, int y);
void handleBoard(int x, int y);
////////检查是否在内部/////////
bool contains(sf::RectangleShape &box, int x, int y);
////////AI移动///////////
std::future<void> work;
int status = 0;
void AIMove();
int main()
{
    mainWindow.create(sf::VideoMode(1500, 1000), L"Heap Overflow 点格棋", sf::Style::Default);
    mainWindow.setVerticalSyncEnabled(true);
    gameBoard = new Board;
    font.loadFromFile("res/LXGWWenKai-Bold.ttf");
    texture.loadFromFile("res/board.jpg", sf::IntRect(0, 0, 1000, 1000));
    sprite.setTexture(texture);
    std::ifstream config("config.ini");
    if (config.is_open())
    {
        BlackName.clear();
        WhiteName.clear();
        config >> BlackName >> WhiteName;
        config.close();
    }
    initSidebar();
    while (mainWindow.isOpen())
    {
        sf::Event event;
        while (mainWindow.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
            {
                mainWindow.close();
            }
            else if (event.type == sf::Event::MouseButtonPressed && !work.valid())
            {
                if (event.mouseButton.x < 1000)
                {
                    handleBoard(event.mouseButton.x, event.mouseButton.y);
                }
                else
                {
                    handleButtons(event.mouseButton.x, event.mouseButton.y);
                }
            }
        }
        // cout<<"steps:"<<steps.size()<<endl;
        if (game_begin == 1)
            AIMove();
        mainWindow.clear(sf::Color(66, 66, 66));
        showGameBoard();
        showSidebar();
        mainWindow.display();
    }
}

////////显示左侧棋盘函数//////////
void showGameBoard()
{
    mainWindow.draw(sprite);
    showDots();
    showVisualLine();
    showLine();
    showBox();
}
void showDots()
{
    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 6; ++j)
        {
            sf::CircleShape c(15);
            c.setFillColor(sf::Color::Black);
            c.setPosition(110 + i * 150, 110 + j * 150);
            mainWindow.draw(c);
        }
    }
}
void showVisualLine()
{
    sf::RectangleShape line(sf::Vector2f(120.f, 10.f));
    sf::Vector2i pos = sf::Mouse::getPosition(mainWindow);
    int x = pos.x;
    int y = pos.y;
    line.setFillColor(sf::Color(97, 97, 97));
    if (x < 890 && x > 110 && y < 890 && y > 110)
    {
        if ((y - 110) % 150 < 30 && (x - 140) % 150 < 120)
        {
            line.setPosition((x - 110) / 150 * 150 + 140, (y - 110) / 150 * 150 + 120);
            mainWindow.draw(line);
        }
        else if ((x - 110) % 150 < 30 && (y - 140) % 150 < 120)
        {
            line.rotate(90.f);
            line.setPosition((x - 110) / 150 * 150 + 130, (y - 110) / 150 * 150 + 140);
            mainWindow.draw(line);
        }
    }
}
void showLine()
{
    for (int i = 0; i < 11; i += 2)
    {
        for (int j = 1; j < 11; j += 2)
        {
            if (gameBoard->map[i][j] == OCCLINE)
            {
                sf::RectangleShape line(sf::Vector2f(120.f, 10.f));
                line.setFillColor(sf::Color::Black);
                line.setPosition(j / 2 * 150 + 140, i / 2 * 150 + 120);
                mainWindow.draw(line);
            }
        }
    }
    for (int i = 1; i < 11; i += 2)
    {
        for (int j = 0; j < 11; j += 2)
        {
            if (gameBoard->map[i][j] == OCCLINE)
            {
                sf::RectangleShape line(sf::Vector2f(10.f, 120.f));
                line.setFillColor(sf::Color::Black);
                // 棋盘的x对应y，棋盘的y对应x
                line.setPosition(j / 2 * 150 + 120, i / 2 * 150 + 140);
                mainWindow.draw(line);
            }
        }
    }
    if (nowMove.first != -1)
    {
        if (nowMove.first % 2 == 0) // 横线
        {
            sf::RectangleShape line(sf::Vector2f(120.f, 2.f));
            line.setFillColor(sf::Color::Red);
            line.setPosition(nowMove.second / 2 * 150 + 140, nowMove.first / 2 * 150 + 124);
            mainWindow.draw(line);
        }
        else // 竖线
        {
            sf::RectangleShape line(sf::Vector2f(2.f, 120.f));
            line.setFillColor(sf::Color::Red);
            line.setPosition(nowMove.second / 2 * 150 + 124, nowMove.first / 2 * 150 + 140);
            mainWindow.draw(line);
        }
    }
}
void showBox()
{
    for (int i = 1; i < 11; i += 2)
    {
        for (int j = 1; j < 11; j += 2)
        {
            if (gameBoard->map[i][j] != EMPTY)
            {
                sf::RectangleShape box(sf::Vector2f(120.f, 120.f));
                if (gameBoard->map[i][j] == BLACK)
                {
                    box.setFillColor(sf::Color(244, 67, 54));
                }
                else
                {
                    box.setFillColor(sf::Color(33, 150, 243));
                }
                box.setPosition(j / 2 * 150 + 140, i / 2 * 150 + 140);
                mainWindow.draw(box);
            }
        }
    }
}
/////////显示右侧信息和按钮/////////
void initSidebar()
{
    ///////显示当前玩家////////
    now_player.setFillColor(sf::Color(158, 158, 158));
    now_player.setOutlineColor(sf::Color(183, 28, 28));
    now_player.setOutlineThickness(10);
    //////玩家切换////////
    first_button.setPosition(1100, 250);
    second_button.setPosition(1300, 250);
    //////游戏开始////////
    GameButton.setPosition(1100, 350);
    //////undo和redo//////
    undo_button.setFillColor(sf::Color(41, 121, 255));
    redo_button.setFillColor(sf::Color(29, 233, 182));
    undo_button.setPosition(1050, 500);
    redo_button.setPosition(1300, 500);
    //////打印棋盘////////
    print_button.setFillColor(sf::Color(121, 85, 72));
    print_button.setPosition(1100, 650);
    ///////加载棋盘////////
    load_button.setFillColor(sf::Color(69, 90, 100));
    load_button.setPosition(1100, 800);
}
void showSidebar()
{
    showInformation();
    showNowPlayer();
    showSetPlayer();
    showGameBegin();
    showUndoAndRedo();
    showPrintBoard();
    showLoadBoard();
}
void showNowPlayer()
{
    if (nowPlayer == BLACK)
    {
        now_player.setPosition(1050, 50);
    }
    else
    {
        now_player.setPosition(1430, 50);
    }
    mainWindow.draw(now_player);
}
void showInformation()
{
    ////////先后手/////////////
    sf::Text x_title(L"先手", font, 50);
    sf::Text h_title(L"后手", font, 50);
    x_title.setPosition(1100, 25);
    h_title.setPosition(1300, 25);
    mainWindow.draw(x_title);
    mainWindow.draw(h_title);
    ////////////得分//////////////
    sf::Text x_score(std::to_string(gameBoard->blackBox), font, 40);
    sf::Text h_score(std::to_string(gameBoard->whiteBox), font, 40);
    x_score.setPosition(1130, 80);
    h_score.setPosition(1330, 80);
    mainWindow.draw(x_score);
    mainWindow.draw(h_score);
    ////////显示时间/////////
    std::string tem;
    time_t b_time = black_time.get();
    if (b_time > 60)
    {
        tem = std::to_string(b_time / 60) + "m " + std::to_string(b_time % 60) + "s";
    }
    else
    {
        tem = "" + std::to_string(b_time) + "s";
    }
    sf::Text x_time(tem, font, 30);
    tem.clear();
    time_t w_time = white_time.get();
    if (w_time > 60)
    {
        tem = std::to_string(w_time / 60) + "m " + std::to_string(w_time % 60) + "s";
    }
    else
    {
        tem = std::to_string(w_time) + "s";
    }
    sf::Text h_time(tem, font, 30);
    x_time.setPosition(1130, 150);
    h_time.setPosition(1330, 150);
    mainWindow.draw(x_time);
    mainWindow.draw(h_time);
}
void showSetPlayer()
{
    sf::Text first_human_text(L"玩家", font, 40);
    sf::Text first_ai_text(L"机器", font, 40);
    sf::Text second_human_text(L"玩家", font, 40);
    sf::Text second_ai_text(L"机器", font, 40);
    first_human_text.setPosition(1110, 260);
    first_ai_text.setPosition(1110, 260);
    second_human_text.setPosition(1310, 260);
    second_ai_text.setPosition(1310, 260);

    if (black_ai)
    {
        first_button.setFillColor(sf::Color(1, 87, 155));
        mainWindow.draw(first_button);
        mainWindow.draw(first_ai_text);
    }
    else
    {
        first_button.setFillColor(sf::Color(255, 196, 0));
        mainWindow.draw(first_button);
        mainWindow.draw(first_human_text);
    }
    if (white_ai)
    {
        second_button.setFillColor(sf::Color(1, 87, 155));
        mainWindow.draw(second_button);
        mainWindow.draw(second_ai_text);
    }
    else
    {
        second_button.setFillColor(sf::Color(255, 196, 0));
        mainWindow.draw(second_button);
        mainWindow.draw(second_human_text);
    }
}
void showGameBegin()
{
    sf::Text BeginText("", font, 50);

    BeginText.setPosition(1150, 370);
    if (game_begin == 0)
    {
        BeginText.setString(L"开始游戏");
        GameButton.setFillColor(sf::Color(76, 175, 80));
    }
    else if (game_begin == 1)
    {
        BeginText.setString(L"结束游戏");
        GameButton.setFillColor(sf::Color(183, 28, 28));
    }
    else
    {
        BeginText.setString(L"继续游戏");
        GameButton.setFillColor(sf::Color(255, 152, 0));
    }
    mainWindow.draw(GameButton);
    mainWindow.draw(BeginText);
}
void showUndoAndRedo()
{
    sf::Text undo_text("undo", font, 50);
    sf::Text redo_text("redo", font, 50);
    undo_text.setPosition(1070, 510);
    redo_text.setPosition(1325, 510);
    mainWindow.draw(undo_button);
    mainWindow.draw(redo_button);
    mainWindow.draw(undo_text);
    mainWindow.draw(redo_text);
}
void showPrintBoard()
{
    sf::Text print_text(L"打印棋盘", font, 50);
    print_text.setPosition(1150, 675);
    mainWindow.draw(print_button);
    mainWindow.draw(print_text);
}
void showLoadBoard()
{
    sf::Text load_text(L"加载棋局", font, 50);
    load_text.setPosition(1150, 825);
    mainWindow.draw(load_button);
    mainWindow.draw(load_text);
}
////////处理按钮//////////
void handleButtons(int x, int y)
{
    if (contains(first_button, x, y))
    {
        black_ai = !black_ai;
    }
    else if (contains(second_button, x, y))
    {
        white_ai = !white_ai;
    }
    else if (contains(GameButton, x, y))
    {
        if (game_begin == 1)
        {
            delete gameBoard;
            gameBoard = new Board;
            game_begin = 0;
            nowPlayer = BLACK;
            black_time.reset();
            white_time.reset();
            top = -1;
            nowMove = {-1, -1};
        }
        else if (game_begin == 0)
        {
            game_begin = 1;
            if (nowPlayer == BLACK)
            {
                black_time.begin();
                white_time.begin();
                white_time.stop();
            }
            else
            {
                white_time.begin();
                black_time.begin();
                black_time.stop();
            }
        }
        else
        {
            game_begin = 1;
            if (nowPlayer == BLACK)
            {
                black_time.start();
            }
            else
            {
                white_time.start();
            }
        }
    }
    else if (contains(redo_button, x, y))
    {
        black_time.stop();
        white_time.stop();
        if (top != 60 && steps[top + 1].player != EMPTY)
        {
            int temp = steps[top + 1].player;
            while (top != 60 && steps[top + 1].player != EMPTY && steps[top + 1].player == temp)
            {
                top++;
                gameBoard->move(steps[top].player, steps[top].action);
                nowMove = steps[top].action;
            }
            nowPlayer = -temp;
        }
        game_begin = -1;
    }
    else if (contains(undo_button, x, y))
    {
        black_time.stop();
        white_time.stop();
        if (top != -1)
        {
            int temp = steps[top].player;
            while (top != -1 && steps[top].player == temp)
            {
                gameBoard->unmove(steps[top].action);
                top--;
            }
            nowMove = steps[top].action;
            nowPlayer = temp;
        }
        game_begin = -1;
    }
    else if (contains(print_button, x, y))
    {
        data_of_game ginfo =
            data_of_game(gameBoard->blackBox, gameBoard->whiteBox, BlackName.c_str(), WhiteName.c_str());
        ginfo.recordstep(steps);
        ginfo.endrecord();
        ginfo.printdata();
        ginfo.deleterecord();
    }
    else if (contains(load_button, x, y))
    {
    }
}
void handleBoard(int x, int y)
{
    if (x > 890 || x < 110 || y > 890 || y < 110)
        return;
    if ((y - 110) % 150 < 30 && (x - 140) % 150 < 120)
    {
        // x坐标对应棋盘的y，y坐标对应棋盘的x
        int bx = (y - 110) / 150 * 2;
        int by = (x - 140) / 150 * 2 + 1;
        if (gameBoard->map[bx][by] == OCCLINE)
        {
            return;
        }
        ////////记录步骤
        steps[++top] = {nowPlayer, LOC{bx, by}};
        nowMove = {bx, by};
        //////占边
        if (gameBoard->move(nowPlayer, {bx, by}) == 0)
        {
            if (game_begin == 1)
            {
                if (nowPlayer == BLACK)
                {
                    black_time.stop();
                    white_time.start();
                }
                else
                {
                    white_time.stop();
                    black_time.start();
                }
            }
            nowPlayer = -nowPlayer;
        }
    }
    if ((x - 110) % 150 < 30 && (y - 140) % 150 < 120)
    {
        // x坐标对应棋盘的y，y坐标对应棋盘的x
        int bx = (y - 140) / 150 * 2 + 1;
        int by = (x - 110) / 150 * 2;
        if (gameBoard->map[bx][by] == OCCLINE)
        {
            return;
        }
        ////////记录步骤
        steps[++top] = {nowPlayer, LOC{bx, by}};
        nowMove = {bx, by};
        //////占边
        if (gameBoard->move(nowPlayer, {bx, by}) == 0)
        {
            if (game_begin == 1)
            {
                if (nowPlayer == BLACK)
                {
                    black_time.stop();
                    white_time.start();
                }
                else
                {
                    white_time.stop();
                    black_time.start();
                }
            }
            nowPlayer = -nowPlayer;
        }
    }
}
////////检查是否在内部/////////
bool contains(sf::RectangleShape &box, int x, int y)
{
    x -= box.getPosition().x;
    y -= box.getPosition().y;
    return box.getLocalBounds().contains(x, y);
}
////////AI下棋函数//////////
void AIMove()
{
    if (!work.valid())
    {
        if (nowPlayer == BLACK)
        {
            if (black_ai)
            {
                ai_steps.clear();
                work = std::async(std::launch::async, gameTurnMove, std::ref(*gameBoard), std::ref(nowPlayer), &status,
                                  std::ref(ai_steps));
            }
        }
        else
        {
            if (white_ai)
            {
                ai_steps.clear();
                work = std::async(std::launch::async, gameTurnMove, std::ref(*gameBoard), std::ref(nowPlayer), &status,
                                  std::ref(ai_steps));
            }
        }
    }
    if (status)
    {
        work.wait();
        work.get();
        if (nowPlayer == BLACK)
        {
            black_time.stop();
            white_time.start();
        }
        else
        {
            white_time.stop();
            black_time.start();
        }
        // cout << "size: " << ai_steps.size() << endl;
        for (auto &i : ai_steps)
        {
            steps[++top] = {nowPlayer, i};
        }
        nowMove = ai_steps.back();
        nowPlayer = -nowPlayer;
        status = 0;
    }
}
