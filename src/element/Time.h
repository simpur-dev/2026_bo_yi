#include <ctime>
#ifndef DOTS_AND_BOXES_TIME_H
#define DOTS_AND_BOXES_TIME_H

class Time
{
  private:
    time_t _begin;
    time_t _clock;

  public:
    void begin();
    void start();
    void stop();
    time_t get() const;
    void reset();
};

#endif // DOTS_AND_BOXES_TIME_H
