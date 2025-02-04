#define _POSIX_C_SOURCE 200112L
//============================================================================================================
// USER AND COMMAND INTERPRETER LIBRARY 
//============================================================================================================

// Standard Libraries
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <allegro.h>
// Custom Libraries Headers
#include "ptask.h"
#include "ball_and_beam_dynamics.h"
#include "command_interpreter.h"
#include "qlearn.h"

// PI [rad] constant definition for math calculations:
#define PI 3.14159265358979323846

//-----------------------------------------------------------------------------------
// USER FUNCTION DEFINITIONS:
//-----------------------------------------------------------------------------------

void get_key_codes(char* scan, char* ascii)
{
int     k;
        if (keypressed()) {

            k = readkey();
            *ascii = k;       // Extract ascii code
            *scan = k >> 8;   // Extract scan code
        }
        else {
            *scan = 0;
            *ascii = 0;
        }    
}
//-----------------------------------------------------------------------------------

void user_init()
{
int       tret;                    // task_create() return flag     
          install_keyboard();      // Allegro keyboard installation

          tret = task_create(3, user_task, COMMAND_PER, COMMAND_DL, COMMAND_PRIO, ACT);
          if (tret == 0) {
                printf("User task created successfully!\n");
          }
          else {
                printf("User task not created successfully!\n");
          }    
}
//-----------------------------------------------------------------------------------

void* user_task(void* arg)
{
int     i;  // task index

        i = get_task_index(arg);
        wait_for_activation(i);          

        while (!is_end()) { 

          command_interpreter();

            // Deadline miss check:
          if (deadline_miss(i)) {
                    printf("The User Task missed the deadline!\n");
          }

          wait_for_period(i); 
        }
        return NULL;
}
//-----------------------------------------------------------------------------------

void command_interpreter()
{    
char           scan, ascii;             // keyboard keycodes buffers
float          delta_theta_ref;         // minimum step of the motor angle
struct status  current_state;           // current state of the ball and beam system
float          delta_f = 0.01;          // friction coefficient increment

               current_state = get_system_status();    
               delta_theta_ref = (float)THETA_STEP * PI / 180;

               // Get the pressed key ascii and scan parts:
               get_key_codes(&scan, &ascii);

               // Performs actions based on the pressed key:
               switch (scan) {
                    case KEY_UP:
                         set_theta_ref(current_state.theta_ref + delta_theta_ref);
                         break;
                    case KEY_DOWN:
                         set_theta_ref(current_state.theta_ref - delta_theta_ref);
                         break;
                    case KEY_LEFT:
                         set_disturbing_force_flag(LEFT_PUSH);
                         break;
                    case KEY_RIGHT:
                         set_disturbing_force_flag(RIGHT_PUSH);
                         break;
                    case KEY_PLUS_PAD:
                         change_friction_coeff(delta_f);
                         break;
                    case KEY_MINUS_PAD:
                         change_friction_coeff(-delta_f);
                         break;                      
                    default: break;     
               }

               switch (ascii) {
                    case 'e':
                         set_rl_mode(EXPLOIT_MODE);
                         printf("Mode : Q-MATRIX EXPLOITATION\n");
                         break;
                    case 'p':
                         set_rl_mode(PLAY_MODE);
                         printf("Mode : USER PLAYING MODE\n");
                         break;
                    case 'r':
                         set_rl_mode(LEARNING_MODE);
                         printf("Mode : REINFORCEMENT LEARNING\n");
                         break;
                    case 'l':
                         set_rl_load_flag(1);
                         break;
                    case 's':
                         set_rl_save_flag(1);
                         break;
                    case 'f':
                         task_set_period(4, RL_PER_FAST);
                         task_set_deadline(4, RL_DL_FAST);
                         set_simulation_speed(FAST);
                         printf("RL TASK PERIOD SET TO 30 ms\n FAST SIMULATION\n");
                         break;
                    case 'n':
                         task_set_period(4, RL_PER);
                         task_set_deadline(4, RL_DL);
                         set_simulation_speed(NORMAL);
                         printf("RL TASK PERIOD SET TO 450 ms\n NORMAL PACE SIMULATION\n");
                         break;
                    case 'q':
                         end_simulation();
                         printf("Q KEY PRESSED! SHUTTING DOWN THE APPLICATION..\n");
                         break;
                    case 'g':
                         set_ql_policy(EPS_GREEDY);
                         printf("Q-Learning policy set to EPS-GREEDY\n");
                         break;
                    case 'b':
                         set_ql_policy(BOLTZMANN);
                         printf("Q-Learning policy set to BOLTZMANN\n");
                         break;                      
                    default: break;          
               }               
}
//-----------------------------------------------------------------------------------
