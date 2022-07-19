/*
  Declaration of Class StopWatch
  Old C version on BSD Unix
  (C) 1992 No.2-3062 N.Fujimoto
*/

#ifndef STOP_WATCH_H

#define STOP_WATCH_H

class StopWatch {
    long cur_tv_sec;
    long cur_tv_usec;
    long start_tv_sec;
    long start_tv_usec; 
public:
	void Reset(void);
	void Start(void);
	void Stop(void);
	double GetTime(void);
};

#endif
