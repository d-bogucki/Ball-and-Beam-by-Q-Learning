#define _POSIX_C_SOURCE 200112L
//-----------------------------------------------------------------------------------
// MAIN FILE:
// This .c file is the main entry of the application. It sets the major parameters
// of the Q-Learning process and then starts all the application tasks.
//-----------------------------------------------------------------------------------

// Standard Library Headers:
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <allegro.h>
#include <time.h>
// Custom Library Headers:
#include "ptask.h"
#include "qlearn.h"
#include "ball_and_beam_dynamics.h"
#include "my_graphics.h"
#include "command_interpreter.h"

//-----------------------------------------------------------------------------------
// MAIN:
//-----------------------------------------------------------------------------------
int main()
{
    srand(time(NULL));                  // initialize random generator
    
    //-------------------------------------------------------------------
    // Q-LEARNING PARAMETERS INITIALIZATION:
    //-------------------------------------------------------------------
    ql_init(MAXSTA, 7);                 // Initialize Q-matrix with given number of states and actions 
    ql_set_learning_rate(0.1);          // Set the "learning rate"
    ql_set_discount_factor(0.9);        // Set the "discount factor"
    ql_set_expl_range(1.0, 0.05);       // Set the "exploration range"
    ql_set_expl_decay(0.99);            // Set the "exploration decay"
    //-------------------------------------------------------------------

    ptask_init(SCHED_FIFO);             // Initialize the tasks with given policy

    system_dynamics_init();             // Initialize the Dynamics and the respective task:

    graphics_init();                    // Initialize the graphics and the respective task:

    user_init();                        // Initialize the user/command task:

    RL_init();                          // Initialize RL task:

    wait_for_task_end(1);               // Wait for dynamic task end
    printf("Dynamics Task terminated\n");

    wait_for_task_end(2);               // Wait for graphics task end
    printf("Graphics Task terminated\n");

    wait_for_task_end(3);               // Wait for user task end
    printf("User Task terminated\n");
    
    wait_for_task_end(4);               // Wait for RL task end
    printf("RL Task terminated, exiting from main()..\n");    

    allegro_exit();                     // Shutting down Allegro

    return 0;
}
