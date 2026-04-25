#include "Time.h"
void Time::begin()
{
    _begin = time(nullptr);
    _clock = 0;
}
void Time::stop()
{
    if (_begin == 0)
        return;

    _clock += time(nullptr) - _begin;
    _begin = 0;
}
void Time::start()
{
    _begin = time(nullptr);
}
time_t Time::get() const
{
    if (_begin == 0)
        return _clock;
    else
        return time(nullptr) - _begin + _clock;
}
void Time::reset()
{
    _begin = 0;
    _clock = 0;
}
