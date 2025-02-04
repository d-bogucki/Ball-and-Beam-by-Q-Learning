#define _POSIX_C_SOURCE 200112L
//-----------------------------------------------------------------------------------
// PTASK LIBRARY
//-----------------------------------------------------------------------------------

// Standard Libraries
#include <time.h>
#include <pthread.h>
#include <semaphore.h>
#include <sched.h>
// Custom Libraries Headers
#include "ptask.h"

//===================================================================================
// Global Data:
//===================================================================================

struct timespec ptask_t0;       // system start time

int ptask_policy;               // scheduler

struct task_par tp[MAX_TASKS];  // task array buffer

//===================================================================================
// Time Management Functions:
//===================================================================================

//-----------------------------------------------------------------------------------
// TIME_COPY()
//-----------------------------------------------------------------------------------
void time_copy(struct timespec *td, struct timespec ts)
{
    td->tv_sec  = ts.tv_sec;
    td->tv_nsec = ts.tv_nsec;
}
//-----------------------------------------------------------------------------------
// TIME_ADD_MS()
//-----------------------------------------------------------------------------------
void time_add_ms(struct timespec *t, int ms)
{
    t->tv_sec  += ms/1000;
    t->tv_nsec += (ms%1000)*1000000;

    if (t->tv_nsec > 1000000000) {
        t->tv_nsec -= 1000000000;
        t->tv_sec += 1;
    }
}
//-----------------------------------------------------------------------------------
// TIME_CMP()
//-----------------------------------------------------------------------------------
int time_cmp(struct timespec t1, struct timespec t2)
{
    if (t1.tv_sec > t2.tv_sec) return 1;
    if (t1.tv_sec < t2.tv_sec) return -1;
    if (t1.tv_nsec > t2.tv_nsec) return 1;
    if (t1.tv_nsec < t2.tv_nsec) return -1;
    return 0;
}
//-----------------------------------------------------------------------------------
// GET_SYSTIME()
//-----------------------------------------------------------------------------------
long get_systime(int unit)
{

struct timespec t;
long tu, mul, div;
switch (unit) {
    case MICRO:     mul = 1000000; div = 1000; break;
    case MILLI:     mul = 1000; div = 1000000; break;
    default:        mul = 1000; div = 1000000; break;
    }    
clock_gettime(CLOCK_MONOTONIC, &t);
tu = (t.tv_sec - ptask_t0.tv_sec)*mul;
tu += (t.tv_nsec - ptask_t0.tv_nsec)/div;

return tu;
}

//===================================================================================
// Task Management Functions:
//===================================================================================

//-----------------------------------------------------------------------------------
// PTASK_INIT()
//-----------------------------------------------------------------------------------
void ptask_init(int policy)
{
    int i;
    ptask_policy = policy;
    clock_gettime(CLOCK_MONOTONIC, &ptask_t0);

    // Initialize activation semaphores:
    for (i = 0; i < MAX_TASKS; i++){
        sem_init(&tp[i].tsem, 0, 0);
    }
}

//-----------------------------------------------------------------------------------
// TASK_CREATE()
//-----------------------------------------------------------------------------------
int task_create(int i, void* (*task)(void *), int period, int drel, int prio, int aflag)
{   
    pthread_attr_t myatt;
    struct sched_param mypar;
    int tret;


    if (i >= MAX_TASKS) return -1;

    tp[i].arg = i;
    tp[i].period = period;
    tp[i].deadline = drel;
    tp[i].priority = prio;
    tp[i].dmiss = 0;

    pthread_attr_init(&myatt);
    pthread_attr_setinheritsched(&myatt, PTHREAD_EXPLICIT_SCHED);
    pthread_attr_setschedpolicy(&myatt, ptask_policy);
    mypar.sched_priority = tp[i].priority;
    pthread_attr_setschedparam(&myatt, &mypar);
    tret = pthread_create(&tp[i].tid, &myatt, task, (void*)(&tp[i])); 

    if (aflag == ACT) task_activate(i);
    return tret;
}
//-----------------------------------------------------------------------------------
// TASK_ACTIVATE()
//-----------------------------------------------------------------------------------
void task_activate(int i)
{
    sem_post(&tp[i].tsem);
}
//-----------------------------------------------------------------------------------
// WAIT_FOR_ACTIVATION()
//-----------------------------------------------------------------------------------
void wait_for_activation(int i)
{
    struct timespec t;

    sem_wait(&tp[i].tsem);
    clock_gettime(CLOCK_MONOTONIC, &t);
    time_copy(&(tp[i].at), t);
    time_copy(&(tp[i].dl), t);
    time_add_ms(&(tp[i].at), tp[i].period);
    time_add_ms(&(tp[i].dl), tp[i].deadline);
}
//-----------------------------------------------------------------------------------
// GET_TASK_INDEX()
//-----------------------------------------------------------------------------------
int get_task_index(void* arg)
{
    struct task_par *tpar;
    tpar = (struct task_par *)arg;
    return tpar->arg;
}
//-----------------------------------------------------------------------------------
// DEADLINE_MISS()
//-----------------------------------------------------------------------------------
int deadline_miss(int i)
{
    struct timespec now;

    clock_gettime(CLOCK_MONOTONIC, &now);

    if (time_cmp(now, tp[i].dl) > 0){
        tp[i].dmiss++;
        return 1;
    }
    return 0;
}
//-----------------------------------------------------------------------------------
// WAIT_FOR_PERIOD()
//-----------------------------------------------------------------------------------
void wait_for_period(int i)
{
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &(tp[i].at), NULL);
    time_add_ms(&(tp[i].at), tp[i].period);
    time_add_ms(&(tp[i].dl), tp[i].period);
}
//-----------------------------------------------------------------------------------
// TASK_SET_PERIOD()
//-----------------------------------------------------------------------------------
void task_set_period(int i, int per)
{
    tp[i].period = per;
}
//-----------------------------------------------------------------------------------
// TASK_SET_DEADLINE()
//-----------------------------------------------------------------------------------
void task_set_deadline(int i, int dline)
{
    tp[i].deadline = dline;
}
//-----------------------------------------------------------------------------------
// TASK_PERIOD()
//-----------------------------------------------------------------------------------
int task_period(int i)
{
    return tp[i].period;
}
//-----------------------------------------------------------------------------------
// TASK_DEADLINE()
//-----------------------------------------------------------------------------------
int task_deadline(int i)
{
    return tp[i].deadline;
}
//-----------------------------------------------------------------------------------
// TASK_DMISS()
//-----------------------------------------------------------------------------------
int task_dmiss(int i)
{
    return tp[i].dmiss;
}
//-----------------------------------------------------------------------------------
// TASK_ATIME()
//-----------------------------------------------------------------------------------
void task_atime(int i, struct timespec *at)
{
    at->tv_sec = tp[i].at.tv_sec;
    at->tv_nsec = tp[i].at.tv_nsec;
}
//-----------------------------------------------------------------------------------
// TASK_ADLINE()
//-----------------------------------------------------------------------------------
void task_adline(int i, struct timespec *dl)
{
    dl->tv_sec = tp[i].dl.tv_sec;
    dl->tv_nsec = tp[i].dl.tv_nsec;
}
//-----------------------------------------------------------------------------------
// WAIT_FOR_TASK_END()
//-----------------------------------------------------------------------------------
void wait_for_task_end(int i)
{
    pthread_join(tp[i].tid, NULL); 
}
