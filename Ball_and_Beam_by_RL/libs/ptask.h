#define _POSIX_C_SOURCE 200112L
//-----------------------------------------------------------------------------------
// Ptask Library Header File:
//-----------------------------------------------------------------------------------

#include <time.h>
#include <pthread.h>
#include <semaphore.h>

//-----------------------------------------------------------------------------------
// Global Constants:
//-----------------------------------------------------------------------------------
#define MAX_TASKS  10

enum {MICRO, MILLI};    
enum {ACT, NO_ACT};
//-----------------------------------------------------------------------------------
// Task Structure: 
//-----------------------------------------------------------------------------------
struct task_par {

    int arg;            // task argument
    long wcet;          // task WCET in us (micro-seconds)
    int period;         // task period in ms
    int deadline;       // relative deadline in ms
    int priority;       // task priority in [0,99]
    int dmiss;          // # of deadline misses
    struct timespec at; // next activation time
    struct timespec dl; // current absolute deadline
    pthread_t tid;      // thread id
    sem_t tsem;         // activation semaphore
};

//-----------------------------------------------------------------------------------
// Time Management Functions Prototypes:
//-----------------------------------------------------------------------------------

/* Time Copy: 
* This function copies a source time variable "ts" in a destination
* variable pointed by "td"
*/
void time_copy(struct timespec *td, struct timespec ts);

/* Time add ms: 
* This function adds a value "ms" expressed in milliseconds to the time
* variable pointed by t
*/
void time_add_ms(struct timespec *t, int ms);

/* Time cmp: 
* This function compares two time variables "t1" and "t2" and returns
* 0 if they are equal, 1 if t1 > t2, -1 if t1 < t2 
*/
int time_cmp(struct timespec t1, struct timespec t2);

/* get systime: 
* This function returns the current elapsed time since "ptask_t0", the global
* variable that stores the start time of the application
* param: "unit" is either "MICRO" or "MILLI" (default is MILLI) 
*/ 
long get_systime(int unit);

//-----------------------------------------------------------------------------------
// Task Management Functions Prototypes:
//-----------------------------------------------------------------------------------

/* ptask init:
* This function initialize all the tasks, stores the application start time in "ptask_t0"
* and initializes all the set of private semaphore for managing explicit activation
* of aperiodic tasks
* param: "policy" , ...
*/
void ptask_init(int policy);

/* Task create:
* This function stores all the specified task parameters in the "task_par" structure
* , set the attributes and call the "pthread_create" function
*/
int task_create(int i, void* (*task)(void *), int period, int drel, int prio, int aflag);

/* Get task index:
* This function retrieves the task index stored in tp->arg
*/
int get_task_index(void* arg);

/* Wait for activation:
* This function check the task's semaphore:
* if unblocked then it updates the next activation time and absolute deadline based
* on the current time
*/
void wait_for_activation(int i);

/* Task activate:
* This function calls the "sem_post" function on the current task's semaphore
*/
void task_activate(int i);

/* Deadline miss:
* This function checks if the current task get past his deadline, if so it does 
* increment the "dmiss" value by one and return 1, otherwise it returns 0
*/
int deadline_miss(int i);

/* Wait for period:
* This function supends the calling task until the next activation and, when awaken
* , updates activation time and deadline
*/
void wait_for_period(int i);

/* Task set period:
* This function changes the period of a task with given index "i"
*/
void task_set_period(int i, int per);

/* Task set period:
* This function changes relative deadline of a task with given index "i"
*/
void task_set_deadline(int i, int dline);

/* Task period:
* This function get the period of a running task
*/
int task_period(int i);

/* Task deadline:
* This function get the relative deadline of a running task
*/
int task_deadline(int i);

/* Task dmiss:
* This function returns the "dmiss" parameter of a task with index "i"
*/
int task_dmiss(int i);

/* Task atime:
* This function copies the activation time of a task with index "i" in the
* timespec struct pointed by "at"
*/
void task_atime(int i, struct timespec *at);

/* Task adline:
* This function copies the absolute deadline of a task with index "i" in the
* timespec struct pointed by "dl"
*/
void task_adline(int i, struct timespec *dl);

/* Wait for task end:
* This function calls the "pthread_join" function on the task with index "i"
*/
void wait_for_task_end(int i);
