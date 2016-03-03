#ifndef _included_timer_h
#define _included_timer_h

#include "Statistics.h"

#define STATS_ITERATION	(0)
#define STATS_DGEMM	(1)
#define STATS_COMM	(2)
#define NUM_STATS	(3)

static const char *StatsNames[NUM_STATS] = 
{
   "Iteration", "dgemm", "Communication"
};

extern Statistics StatMaster;

#define STATS_START(name)	StatMaster.Start(name)
#define STATS_STOP(name)	StatMaster.Stop(name)
#define STATS_RESET()		StatMaster.Reset()
#define STATS_REPORT(name)	StatMaster.Report(name)

#endif
