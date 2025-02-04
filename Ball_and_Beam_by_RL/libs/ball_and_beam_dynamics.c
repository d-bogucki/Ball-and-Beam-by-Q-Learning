#define _POSIX_C_SOURCE 200112L
//============================================================================================================
// BALL AND BEAM DYNAMICS LIBRARY
//============================================================================================================

// Standard libraries:
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
// Custom libraries headers:
#include "ptask.h"
#include "ball_and_beam_dynamics.h"
#include "qlearn.h"

// PI [rad] constant definition for math calculations:
#define PI 3.14159265358979323846

//-----------------------------------------------------------------------------------------------------------
// SYSTEM GLOBAL VARIABLES
//-----------------------------------------------------------------------------------------------------------

static struct status    ball_and_beam_system;                       // System status buffer
const int               g = G0;                                     // Gravity acceleration
static float            friction_coeff = ROLLING_FRICTION_COEFF;    // Default value = 0.05

// End of Simulation Flag:
static int              end = 0;

// Disturbance Force Flag:
static int              disturbing_force_flag = NO_PUSH;

// System buffer Mutex Initialization:
pthread_mutex_t         system_mux = PTHREAD_MUTEX_INITIALIZER;
                       

//-----------------------------------------------------------------------------------------------------------
// SYSTEM FUNCTIONS DEFINITIONS:
//-----------------------------------------------------------------------------------------------------------

float action2theta_ref(int action)
{
// Note: This function assumes the action value is between [0, NUM_ACTIONS-1]     
float   theta_ref, theta_step_rad, theta_min;

        theta_step_rad = THETA_STEP * PI / 180;                     // theta step in [rad]
        theta_min = -((NUM_ACTIONS - 1) / 2) * theta_step_rad;      // theta min in [rad]

        theta_ref = (theta_min + theta_step_rad * action);          // theta_ref in [rad]

return  theta_ref;
}
//-----------------------------------------------------------------------------------------------------------

int theta_ref2action(float theta_ref)
{
int     i;                                                          // Iterator 
float   theta_step_rad, theta_min; 
float   theta_list[NUM_ACTIONS];                                    // theta_ref list = {theta_min, theta_min + theta_step, theta_min + 2*theta_step..} 
float   tolerance = 0.5*PI/180;                                     // 0.5 [deg] tolerance for float comparisons                                      

        theta_step_rad = THETA_STEP * PI / 180;                     // theta step in [rad]
        theta_min = -((NUM_ACTIONS - 1) / 2) * theta_step_rad;      // theta min in [rad

        // Calculating the theta_ref_list:
        for (i=0; i<NUM_ACTIONS; i++) {
            theta_list[i] = (theta_min + theta_step_rad * i);
        }
        // Extracting the corresponding action if the input value matches with one inside the list:
        for (i=0; i <NUM_ACTIONS; i++) {
            if (fabs(theta_ref - theta_list[i]) <= tolerance) {
                return i;
            }
        }
        // If no match was found, printing an error message:
        printf("theta_ref2action() conversion error!, returning 1\n");
        return 1;
}
//-----------------------------------------------------------------------------------------------------------

void ball_and_beam_dynamics()
{
int     i;                                                          // Iterator
int     sim_speed = get_simulation_speed();                         // Current simulation speed
int     dynamic_steps;                                              // Dynamic steps to perform based on simulation speed
// System buffers:
float   x, v, a, theta, omega, alpha, theta_ref;                    // System current state buffers
float   x_new, v_new, a_new, theta_new, omega_new, alpha_new;       // System new state buffers
float   friction_acc;                                               // Acceleration component from friction rolling force
int     s;                                                          // Discretized system state for Q-learning
        
        if (sim_speed == FAST) dynamic_steps = 15;
        else dynamic_steps = 1;

        // Mutex lock:
        pthread_mutex_lock(&system_mux);

        for (i=0; i < dynamic_steps; i++) {
            // Getting the current values from the system global variable: (shorter variables names)
            x = ball_and_beam_system.x;
            v = ball_and_beam_system.v;
            theta = ball_and_beam_system.theta;
            omega = ball_and_beam_system.omega;
            alpha = ball_and_beam_system.alpha;
            theta_ref = ball_and_beam_system.theta_ref;
            //---------------------------------------------------------------------------------------------------
            // EA: (Forward Euler Integration Method)
            //---------------------------------------------------------------------------------------------------
            // Acceleration computation:
            a = ((float)2 / 7 * BALL_RADIUS * alpha + (float)5 / 7 * x * pow(omega,2) - (float)5 / 7 * g * sin(theta));
            friction_acc = (float)5 / 7 * g * cos(theta) * friction_coeff;
            // Rolling Friction contribution to the acceleration: ("sgn() function implementation")
            if (v > 0) {
                a = a - friction_acc;
            }
            if (v < 0) {
                a = a + friction_acc;
            }
            // Next step states:
            x_new = x + v * T_DYNAMICS;
            x_new = x_new + (float)1 / 2 * a * pow(T_DYNAMICS, 2);  // Additional term to increase accuracy of EA integration method

            v_new = v + a * T_DYNAMICS;  

            // Handling ball dynamics when standing still at the beam limit:
            if (x >= (BEAM_LENGTH-BALL_RADIUS - 0.001) && (((v_new) * v) <= 0.0) && (theta < 0.0)) {
            v_new = 0.0;
            }            
            if (x <= (BALL_RADIUS + 0.001) && (((v_new) * v) <= 0.0) && (theta > 0.0)) {    
            v_new = 0.0;
            }
            // Handling ball dynamics when theta angle close to 0:
            if ((((v_new) * v) <= 0.0) && (theta <= 0.0 + 0.01) && (theta >= 0.0 - 0.01)){
                v_new = 0.0;
            }
            // Handling the rolling friction force. It can't change the ball velocity direction!
            if (v>0 && (v * (v + (a - friction_acc) * T_DYNAMICS)) < 0) {
                v_new = 0.0;
                x_new = x_new + (float)1 / 2 * friction_acc * pow(T_DYNAMICS, 2); 
            }
            if (v<0 && (v * (v + (a + friction_acc) * T_DYNAMICS)) < 0) {
                v_new = 0.0;
                x_new = x_new - (float)1 / 2 * friction_acc * pow(T_DYNAMICS, 2);
            }
            // Disturbing Force:
            if (disturbing_force_flag) {
                v_new = v_new + (float)disturbing_force_flag * DISTURBING_VELOCITY;
                disturbing_force_flag = NO_PUSH;
            }
            // Beam Dynamics, given a servo-motor with first order dynamics like below:
            // theta_dot = (-1/tau*theta + K/tau*theta_ref), with K = 1
            theta_new = theta + T_DYNAMICS * (-1/MOTOR_TAU * theta + theta_ref/MOTOR_TAU);
            // Beam angular speed and acceleration raw calculations:
            omega_new = (theta_new - theta)/T_DYNAMICS;
            alpha_new = (omega_new - omega)/T_DYNAMICS;
            a_new = a;        
            // Update of the system states:
            ball_and_beam_system.x = x_new;
            ball_and_beam_system.v =  v_new;
            ball_and_beam_system.a = a_new;
            ball_and_beam_system.theta = theta_new;
            ball_and_beam_system.omega = omega_new;
            ball_and_beam_system.alpha = alpha_new;

            // Handling the Beam Limits:            
            handle_beam_limits();         
            
            // Discretized RL state update:
            s = state_2_state_rl(ball_and_beam_system.x, ball_and_beam_system.v);
            ball_and_beam_system.s = s;
        }
        // Mutex Unlock:
        pthread_mutex_unlock(&system_mux);
}
//-----------------------------------------------------------------------------------------------------------

void handle_beam_limits()
{
// 1D Anelastic Collision:
float           lost_energy_percentage, loss_factor;
                lost_energy_percentage = 40;

        if (ball_and_beam_system.x >= (BEAM_LENGTH - BALL_RADIUS) || ball_and_beam_system.x <= (BALL_RADIUS)) {

            // Handling ball position:
            if (ball_and_beam_system.x >= (BEAM_LENGTH - BALL_RADIUS)) {
                ball_and_beam_system.x = (BEAM_LENGTH - BALL_RADIUS);
            }
            else ball_and_beam_system.x = BALL_RADIUS;
            // Handling ball velocity, after bounce off the beam limit:
            loss_factor = (1 - (float)lost_energy_percentage/100);
            // The ball bounces off in opposite direction with less energy:
            ball_and_beam_system.v = -loss_factor * ball_and_beam_system.v; 
        }
}
//-----------------------------------------------------------------------------------------------------------

void* system_dynamics_task(void* arg){

int     i;      // task index

        i = get_task_index(arg);
        wait_for_activation(i);        

        while (!end) {

            // Make a dynamic step:
            ball_and_beam_dynamics();
    
            // Deadline miss check:
            if (deadline_miss(i)) {
                printf("The Dynamics Task missed the deadline!\n");
            }
            wait_for_period(i);
        }
        return NULL;      
}
//-----------------------------------------------------------------------------------------------------------

void system_dynamics_init()
{
int     tret;    // task_create() return flag

        // System status initialization, ball at the beam center with zero velocity:
        ball_and_beam_system.x = BEAM_LENGTH/2;
        ball_and_beam_system.v = 0.0;
        ball_and_beam_system.a = 0.0;
        ball_and_beam_system.theta = 0.0;
        ball_and_beam_system.omega = 0.0;
        ball_and_beam_system.alpha = 0.0;
        ball_and_beam_system.theta_ref = 0.0;
        ball_and_beam_system.action = theta_ref2action(ball_and_beam_system.theta_ref);
        ball_and_beam_system.s = state_2_state_rl(ball_and_beam_system.x, ball_and_beam_system.v);

        // Dynamics Task creation:
        tret = task_create(1, system_dynamics_task, DYNAMICS_PER, DYNAMICS_DL, DYNAMICS_PRIO, ACT);
        if (tret == 0) {
                printf("Dynamic task created successfully!\n");
        }
        else {
                printf("Dynamic task not created successfully!\n");
        }    
}
//-----------------------------------------------------------------------------------------------------------

struct status get_system_status()
{
struct status   sys_status_buf;     // System buffer

                pthread_mutex_lock(&system_mux);

                sys_status_buf = ball_and_beam_system;

                pthread_mutex_unlock(&system_mux);

    return sys_status_buf;
}
//-----------------------------------------------------------------------------------------------------------

void set_system_status(float x, float v)
{

    pthread_mutex_lock(&system_mux);

    ball_and_beam_system.x = x;
    ball_and_beam_system.v = v;
    ball_and_beam_system.a = 0.0;  
    ball_and_beam_system.theta = 0.0;
    ball_and_beam_system.omega = 0.0;
    ball_and_beam_system.alpha = 0.0;
    ball_and_beam_system.theta_ref = 0.0;
    ball_and_beam_system.s = state_2_state_rl(x, v);
    ball_and_beam_system.action = theta_ref2action(ball_and_beam_system.theta_ref);

    pthread_mutex_unlock(&system_mux);
}
//-----------------------------------------------------------------------------------------------------------

void set_theta_ref(float theta_ref)
{

float   theta_step_rad, theta_min, theta_max;

        theta_step_rad = THETA_STEP * PI / 180;                     // theta step in [rad]
        theta_min = -((NUM_ACTIONS - 1) / 2) * theta_step_rad;      // theta min in [rad]
        theta_max = ((NUM_ACTIONS - 1) / 2) * theta_step_rad;       // theta max in [rad]
   
        ball_and_beam_system.theta_ref = theta_ref;
        // Set theta to a maximum or minimum value:
        if (theta_ref >= theta_max) ball_and_beam_system.theta_ref = theta_max;
        if (theta_ref <= theta_min) ball_and_beam_system.theta_ref = theta_min;  

}
//-----------------------------------------------------------------------------------------------------------

void end_simulation()
{
    end = 1;
}
//-----------------------------------------------------------------------------------------------------------

int is_end()
{
    return end;
}
//-----------------------------------------------------------------------------------------------------------

rl_state_pair get_rl_state_pair(int s)
{
rl_state_pair   rl_pair;                                        // (s_x, s_v) discretized pair buffer
int             s_v, s_x;                                       // s_x in [1, n_states_x], s_v in [1, n_states_v]
int             n_states, n_states_x, n_states_v;

                n_states = get_n_states();                      // Getting the current n_states values from "qlearn" library buffer

                // Hp. It is assumed that n_states_x = 3/2*n_states_v, hence (n_states_x)*(n_states_v) = n_states
                n_states_v = floor(sqrt((float)2 / 3 * n_states));  
                n_states_x = (float)n_states / n_states_v;        
                if (n_states_x*n_states_v != n_states) printf("get_rl_state_pair() function ERROR!\n");                                         

                // OBS: The RL assumes s in [0, NUM_STATES-1], The following steps instead assumes it to be in [1, NUM_STATES]
                s = s + 1;

                s_v = (int)floor((float)s / n_states_x) + 1; 
                if (s%n_states_x == 0) s_v = s_v - 1;   

                s_x = (s%n_states_x);   
                if (s%n_states_x == 0) s_x = n_states_x; 

                rl_pair.s_x = s_x;
                rl_pair.s_v = s_v;
                return rl_pair;
}
//-----------------------------------------------------------------------------------------------------------

int state_2_state_rl(float x, float v)
{
int     i;                                          // Iterator
int     n_states, n_states_x, n_states_v;           // Number of discretized states
int     s_x, s_v, s;                                // RL discretized states
float   delta_x;                                    // discretized beam minimum interval [m]
        n_states = get_n_states();                  // Getting the current n_states values from "qlearn" library buffer

        // Hp. It is assumed that n_states_x = 3/2*n_states_v, hence (n_states_x)*(n_states_v) = n_states
        n_states_v = floor(sqrt((float)2 / 3 * n_states));
        n_states_x = (float)n_states / n_states_v;
        
        // Beam lenght is divided in uniform intervals:
        delta_x = (float)BEAM_LENGTH / n_states_x;

        // Compute s_x given the ball position: (s_x in [1, n_states_x])
        for (i=1; i<= n_states_x; i++) {

            if (x >= ((i - 1) * delta_x) && x < (i * delta_x)) {
                s_x = i;
                break;
            }
        }

        // Compute s_v given the ball velocity: (s_v in [1, n_states_v])
        // Note: Up to now this function works ONLY with n_states_v = 10! (hence n_states = 150!)        
        if (v >= 0.0 && v < 0.05)        s_v = 1;
        if (v >= 0.05 && v < 0.1)        s_v = 2;
        if (v >= 0.1 && v < 0.2)         s_v = 3;
        if (v >= 0.2 && v < 0.4)         s_v = 4;
        if (v >= 0.4)                    s_v = 5;
        if (v < 0.0 && v >= -0.05)       s_v = 6;
        if (v < -0.05 && v >= -0.1)      s_v = 7;
        if (v < -0.1 && v >= -0.2)       s_v = 8;
        if (v < -0.2 && v >= -0.4)       s_v = 9;
        if (v < -0.4)                    s_v = 10;        

        // Encoding s_x and s_v into a unique s discretized state: 
        s = s_x + (s_v - 1) * n_states_x;   // s in [1, n_states]
        s = s - 1;                          // s in [0, n_states-1]            
        return s;                             
}
//-----------------------------------------------------------------------------------------------------------

void set_disturbing_force_flag(int k)
{
    disturbing_force_flag = k;
}
//-----------------------------------------------------------------------------------------------------------
void change_friction_coeff(float delta_f)
{
    friction_coeff = friction_coeff + delta_f;
    if (friction_coeff >= 0.5) friction_coeff = 0.5;    // Maximum friction coefficient
    if (friction_coeff <= 0)   friction_coeff = 0.0;    // Minimum friction coefficient

}
//-----------------------------------------------------------------------------------------------------------
float get_friction_coeff()
{
    return friction_coeff;
}
//-----------------------------------------------------------------------------------------------------------
