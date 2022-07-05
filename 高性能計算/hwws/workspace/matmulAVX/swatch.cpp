/*
  Definition of Class StopWatch
  Old C Version on BSD Unix
  (C) 1992 No.2-3062 N.Fujimoto
*/

#include <stdio.h>
#include <sys/time.h>
#include "swatch.h"

//struct timeval {
//         long    tv_sec;         /* seconds */
//         long    tv_usec;        /* and microseconds */
//};

//struct timezone {
//        int     tz_minuteswest; /* minutes west of Greenwich */
//        int     tz_dsttime;     /* type of dst correc */
//};

void StopWatch::Reset(void)
{
    this->cur_tv_sec = 0;
    this->cur_tv_usec = 0;
}

void StopWatch::Start(void)
{
    struct timeval tv;
    struct timezone tz;

    gettimeofday(&tv, &tz);
    this->start_tv_sec = tv.tv_sec;
    this->start_tv_usec = tv.tv_usec; 
}

void StopWatch::Stop(void)
{
    struct timeval tv;
    struct timezone tz;
    long end_tv_sec;
    long end_tv_usec;

    gettimeofday(&tv, &tz);
    end_tv_sec = tv.tv_sec;
    end_tv_usec = tv.tv_usec;

    end_tv_sec -= this->start_tv_sec;
    end_tv_usec -= this->start_tv_usec;
    if (end_tv_usec < 0)
        {
        end_tv_sec--;
        end_tv_usec += 1000000;
	}
    this->cur_tv_sec += end_tv_sec;
    this->cur_tv_usec += end_tv_usec; 
}

double StopWatch::GetTime(void)
{
    return this->cur_tv_sec + 0.000001 * this->cur_tv_usec;
}
