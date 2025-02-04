#define _POSIX_C_SOURCE 200112L
//============================================================================================================
// Q-LEARNING CUSTOM LIBRARY
//============================================================================================================

// Standard Libraries:
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>

// Custom Libraries Headers:
#include "ptask.h"
#include "qlearn.h"
#include "my_graphics.h"
#include "ball_and_beam_dynamics.h"

// PI [rad] constant definition for math calculations:
#define PI 3.14159265358979323846

//------------------------------------------------------------------------------------------------------------
// Q-LEARN GLOBAL VARIABLES
//------------------------------------------------------------------------------------------------------------
static int      nsta;                   // number of states
static int      nact;                   // number of actions

static float    alpha;                  // learning rate
static float    gam;                    // discount factor
static float    decay;                  // decay rate for epsilon
static float    eps_norm = 1.0;         // normal expl. probability
static float    eps_ini;                // initial expl. probability
static float    eps_fini;               // final expl. probability
static float    epsilon;                // current expl. probability

static float    beta_boltz = 0.0;       // Beta coefficient for Boltzmann softmax distribution
static int      ql_policy = EPS_GREEDY; // default Q-Learning Policy

static int      stall_condition_flag = 0;       // 1 if the ball is in stall condition in the beam middle, 0 otherwise

// Q-Learning Matrix:
static int Q[MAXSTA][MAXACT];           // Q Matrix Buffer

// RL Circular Buffer of previous states:
static circular_buffer rl_cbuffer;

// Other flags for different RL task modes: 
static int      rl_mode = PLAY_MODE;
static int      rl_load_flag = 0;               // Load flag (if 1 load the saved Q-matrix from .txt file) 
static int      rl_save_flag = 0;               // Save flag (if 1 save the Q matrix to the .txt file)
static int      simulation_speed = NORMAL;      // NORMAL or FAST simulation modes

//------------------------------------------------------------------------------------------------------------
// Q-LEARNING FUNCTIONS DEFINITIONS:
//------------------------------------------------------------------------------------------------------------

void ql_init(int ns, int na)
{
int     s, a;

        nsta = ns;  
        nact = na;

        if (nsta > MAXSTA) {
            printf("Number of states too big\n");
            exit(1);
        }

        if (nact > MAXACT) {
            printf("Number of actions too big\n");
            exit(1);
        }
        alpha = ALPHA0;         // initial learning rate
        gam = GAMMA0;           // discount factor
        eps_ini = EPSINI;       // intial expl. probability
        eps_fini = EPSFIN;      // final expl. probability
        decay = DECAY0;         // decay rate

        // Q-Matrix Initialization:
        for (s=0; s<nsta; s++) {  
            for (a=0; a<nact; a++)
                Q[s][a] = 0;
        }
        // Circular buffer initialization:
        rl_buffer_init();
}
//------------------------------------------------------------------------------------------------------------

void ql_reduce_exploration()
{
    eps_norm = decay*eps_norm;
    epsilon = eps_fini + eps_norm*(eps_ini - eps_fini);
}

//------------------------------------------------------------------------------------------------------------

float ql_maxQ(int s)
{
int     a;
float   m;

        m = Q[s][0];    // init. with Q value for action 0

        for (a=1; a<nact; a++)
            if (Q[s][a] > m) m = Q[s][a];

        return m;
}
//------------------------------------------------------------------------------------------------------------

float ql_best_action(int s)
{
int     a, ba;
float   m;
        m = Q[s][0];
        ba = 0;
        for (a=1; a<nact; a++)
            if (Q[s][a] > m) {
                m = Q[s][a];
                ba = a;
            }
        return ba;
}
//------------------------------------------------------------------------------------------------------------

float frand(float xmi, float xma)
{
float   r;
        r = rand()/(float)RAND_MAX;             // rand in [0,1)
        return xmi + (xma - xmi)*r;
}
//------------------------------------------------------------------------------------------------------------

int ql_egreedy_policy(int s)
{
int     ra, ba;
float   x;

        ba = ql_best_action(s);
        ra = round(frand(0, 1)*(nact-1));       // random action

        x = frand(0, 1);   

        if (x < epsilon)
                return ra;
        else    return ba;      
}
//------------------------------------------------------------------------------------------------------------

int ql_boltzmann_policy(int s)
{
int     i;                      // Iter
float   m;                      // Max Q[s][a] for a given s
float   sum = 0.0;              // Softmax distribution sum at the denominator
float   p[nact];                // Discrete Probability distribution buffer
float   x;                      // Random number between [0, 1]
int     action = -1;            // Selected action
float   cumulative_prob = 0.0;  // CDF cumulated probability buffer

        m = ql_maxQ(s);

        // sum calculation:
        for (i=0; i < nact; i++) {

                sum = sum + exp(beta_boltz * (Q[s][i] - m));
        }
        // compute the discrete probability for every Q[s][i] (p[i]):
        for (i=0; i < nact; i++) {

                p[i] = exp((beta_boltz * (Q[s][i] - m) - log(sum)));
        }
        // Computing the CDF (Cumulative Distribution Probability) for selection an action
        x = frand(0, 1);
        for (i=0; i < nact; i++) {

                cumulative_prob = cumulative_prob + p[i];
                if (x < cumulative_prob) {
                        action = i;
                        break;
                }
        }
        if (action == -1) {
                printf("Boltzmann action selection ERROR! returning action = 0\n");
                action = 0;
        }
        
        return action;
}
//------------------------------------------------------------------------------------------------------------

float ql_updateQ(int s, int a, int r, int snew)
{
float   Qtarget;      // target Q value
float   TDerr;        // TD error

        Qtarget = r + gam*ql_maxQ(snew);
        TDerr = (Qtarget - Q[s][a]);
        Q[s][a] = Q[s][a] + alpha*TDerr;
        return fabs(TDerr);
}
//------------------------------------------------------------------------------------------------------------

float learn_episode(int s0, int i, int episode_counter)
{
struct status system_status;            // System Status buffer       
int     s, a, r, s_new, steps = 0;      
float   E = 0;                          // TD-Error initialization
        s = s0;
        rl_buffer_insert(s);            // Insert the episode starting state into the circular buffer

        while (steps < MAXSTEPS && !is_end()) {

                // Load Q-Matrix check:
                if (rl_load_flag == 1) {
                        printf("Loading the Q-matrix from the .txt file\n");
                        load_Q_matrix();
                        set_q_table_graphics_flag(1);                   // Set the graphics load flag to 1
                        episode_counter = 1;
                        steps = 0;
                        rl_load_flag = 0;
                }
                // Save Q-Matrix check:
                if (rl_save_flag == 1) {
                        printf("Saving the Q-matrix to the .txt file\n");
                        save_Q_matrix();
                        rl_save_flag = 0;
                }

                if (rl_mode == LEARNING_MODE) {

                        steps++;
                        // Select an action based on the current state and the policy:
                        if (ql_policy == EPS_GREEDY) {
                                a = ql_egreedy_policy(s);               
                        }
                        if (ql_policy == BOLTZMANN) {
                                a = ql_boltzmann_policy(s);
                        }
                        set_theta_ref(action2theta_ref(a));             // Give the action reference to the servo motor

                        // Deadline miss check:
                        if (deadline_miss(i)) {
                                printf("The RL Task missed the deadline!\n");
                        }
                        wait_for_period(i);                             // Wait for the next period of the RL task
                        //--------------------------------------------------------------------------------------------  
                                     
                        system_status = get_system_status();            // Get the current system status      
                        s_new = system_status.s;                        // Update the new state            
                        rl_buffer_insert(s_new);                        // Save the state into the rl circular buffer
                        r = get_reward();                               // Get the current reward:           
                        E += ql_updateQ(s, a, r, s_new);                // Update Q matrix and return the TD error

                        // Update the current RL states for the Graphics task:
                        set_graphics_RL_state(s, a, r, steps, episode_counter, E/steps);                                  
                        s = s_new;                                      // Update the old state    
                        // Reduce the random exploration factor after a given number of steps
                        if (steps%1000 == 0 && ql_policy == EPS_GREEDY) { 
                                ql_reduce_exploration();
                        }
                        // Generate a random position if the ball is in stall condition at the center:
                        if (ball_stall_condition()) {
                                s = generate_random_position();
                                rl_buffer_insert(s);        
                        }  
                }  

                if (rl_mode == EXPLOIT_MODE) {
                        a = ql_best_action(s);                          // Exploit the current Q-matrix
                        set_theta_ref(action2theta_ref(a));
                        // Deadline miss check:
                        if (deadline_miss(i)) {
                                printf("The RL Task missed the deadline!\n");
                        }             
                        wait_for_period(i);
                        //--------------------------------------------------------------------------------------------
                        system_status = get_system_status();
                        s_new = system_status.s;
                        r = 0;                                          // Dummy reward
                        set_graphics_RL_state(s, a, r, steps, episode_counter, E/steps); 
                        s = s_new;
                }
                if (rl_mode == PLAY_MODE) {
                        // Dummy RL loop:
                        if (steps != 0) {
                                steps = steps;
                        }
                        else steps = 1;
                        system_status = get_system_status();
                        a = theta_ref2action(system_status.theta_ref);  // Get the action from the system state buffer
                        r = 0;
                        set_graphics_RL_state(system_status.s, a, r, steps, episode_counter, E/steps);
                        // Deadline miss check:
                        if (deadline_miss(i)) {
                                printf("The RL Task missed the deadline!\n");
                        }
                        wait_for_period(i);
                        //--------------------------------------------------------------------------------------------       
                }
        }
        return E/steps;     // average accumulated TD error
}
//------------------------------------------------------------------------------------------------------------

void qlearn(float eps, int i)
{
float   E;                              // average TD error over an episode
int     s0;                             // initial state
int     episode_counter = 0;            // episode counter
        epsilon = eps;

        do {
                episode_counter++;                      // Increase the episode number
                if (ql_policy == BOLTZMANN) {
                        update_beta_boltz(episode_counter);    
                }
                if (episode_counter == 1) {
                        // In the first episode start in the middle position with zero velocity:
                        s0 = state_2_state_rl(BEAM_LENGTH/2, 0.0);
                }
                else {
                        s0 = generate_random_position();
                }
                // Enter the episode learning loop:
                E = learn_episode(s0, i, episode_counter);
                // Update the average TD error for graphics after every 10 episodes and after the first:
                if (episode_counter % 10 == 0 || episode_counter == 2) {
                        set_avg_episode_TD_err(E);
                }
                // Save the Q-matrix after the full learning loop:
                if (episode_counter == MAXEPI) {
                        printf("Saving the Q-matrix to the .txt file\n");
                        save_Q_matrix();
                }

        } while (episode_counter < MAXEPI && !is_end());

}
//------------------------------------------------------------------------------------------------------------

void*   RL_task(void* arg)
{
int     i;                      // task index
float   eps = eps_ini;          // Initial exploration probability

        i = get_task_index(arg);
        wait_for_activation(i);

        qlearn(eps, i);         // Initialize and start the RL loop

        return NULL;                        
}
//------------------------------------------------------------------------------------------------------------

void RL_init()
{
int     tret;        
        tret = task_create(4, RL_task, RL_PER, RL_DL, RL_PRI, ACT);
        if (tret == 0) {
                printf("RL task created successfully!\n");
        }        
        else {
                printf("RL tasks NOT CREATED successfully! error code: %d\n", tret);
        }        
}
//------------------------------------------------------------------------------------------------------------

float ql_get_learning_rate()
{
float   lr;

        lr = alpha;
        return lr;
}
//------------------------------------------------------------------------------------------------------------

float ql_get_discount_factor()
{
float   df;
        df = gam;
        return df;
}
//------------------------------------------------------------------------------------------------------------

float ql_get_expl_decay()
{
float   ed;
        ed = decay;
        return ed;
}
//------------------------------------------------------------------------------------------------------------

float ql_get_epsilon()
{
float   eps;
        eps = epsilon;
        return eps;
}
//------------------------------------------------------------------------------------------------------------

void ql_set_learning_rate(float lr)
{
        alpha = lr;
}
//------------------------------------------------------------------------------------------------------------

void ql_set_discount_factor(float df)
{
        gam = df;
}
//------------------------------------------------------------------------------------------------------------

void ql_set_expl_range(float eini, float efin)
{
        eps_ini = eini;
        eps_fini = efin;
}
//------------------------------------------------------------------------------------------------------------

void ql_set_expl_decay(float d)
{
        decay = d;
}
//------------------------------------------------------------------------------------------------------------
// OTHER CUSTOM FUNCTIONS:
//------------------------------------------------------------------------------------------------------------

int get_n_states()
{
        return nsta;
}
//------------------------------------------------------------------------------------------------------------

int get_n_actions()
{
        return nact;
}
//------------------------------------------------------------------------------------------------------------

int get_Q_value(int state, int action)
{
        return Q[state][action];
}
//------------------------------------------------------------------------------------------------------------

int get_reward()
{
int             i;
int             r;                                      // reward
int             n_states, n_states_x, n_states_v;       // discretized states
int             s_x_middle;                             // s_x value at the beam center
rl_state_pair   rl_pair_buf[RL_BUF_DIM];                // array of discretized rl pairs (s_x, s_v)

                n_states = get_n_states();
                n_states_v = floor(sqrt((float)2/3*n_states));
                n_states_x = (float)n_states/n_states_v;

                s_x_middle = round((float)n_states_x/2);                              

                // Getting the previous states from the circular buffer:
                for (i=0; i < RL_BUF_DIM; i++) {

                        rl_pair_buf[i] = get_rl_state_pair(rl_buffer_get(i));
                }

                //--------------------------------------------------------------------------------------------
                // POSITIVE REWARDS:
                //--------------------------------------------------------------------------------------------

                // 1) Ball stays in the middle sector for three consecutive RL steps and in the last two has minimum velocity:                
                if ((rl_pair_buf[0].s_x == s_x_middle) && (rl_pair_buf[1].s_x == s_x_middle) 
                        && (rl_pair_buf[2].s_x == s_x_middle) && (rl_pair_buf[0].s_v == 1 || rl_pair_buf[0].s_v == 6)
                        && (rl_pair_buf[0].s_v == rl_pair_buf[1].s_v)) {
                        
                        r = +60;
                        return r;
                }
                

                // 2) Ball in the previous step was not in the middle sector and now is in the middle with minimum velocity:
                if ((rl_pair_buf[1].s_x != s_x_middle) && (rl_pair_buf[0].s_x == s_x_middle) 
                        && ((rl_pair_buf[0].s_v == 1) || (rl_pair_buf[0].s_v == 6))) {
                        r = +30;
                        return r;
                }

                //--------------------------------------------------------------------------------------------
                // NEGATIVE REWARDS:
                //--------------------------------------------------------------------------------------------

                // 1.a) Ball in the previous step wasn't at the right terminal sector, but now is in the terminal sector
                //      and the velocity did not decreased; ball has same speed direction in both steps
                if ((rl_pair_buf[0].s_x == n_states_x)                                  
                        && (rl_pair_buf[1].s_x < n_states_x)                            
                        && (rl_pair_buf[0].s_v <= 5 && rl_pair_buf[1].s_v <= 5)         
                        && (rl_pair_buf[0].s_v >= rl_pair_buf[1].s_v)) {

                        r = - 60;
                        return r;                
                }
                // 1.b) Ball in the previous step wasn't at the left terminal sector, but now is in the terminal sector
                //      and the velocity did not decreased; ball has same speed direction in both steps
                if( (rl_pair_buf[0].s_x == 1)                                         
                        && (rl_pair_buf[1].s_x > 1)                                     
                        && (rl_pair_buf[0].s_v > 5 && rl_pair_buf[1].s_v > 5)       
                        && (rl_pair_buf[0].s_v >= rl_pair_buf[1].s_v)) {

                        r = - 60;
                        return r;                
                }

                // 2.a) Same as 1.a) but the ball has opposite direction between the steps, there was a bounce off the wall
                if ((rl_pair_buf[0].s_x == n_states_x)                                   
                        && (rl_pair_buf[1].s_x < n_states_x)                             
                        && (rl_pair_buf[0].s_v > 6 && rl_pair_buf[1].s_v <= 3)) {        
                                                                                         
                        r = -60;
                        return r;                
                }

                // 2.b) Same as 1.b) but the ball has opposite direction between the steps, there was a bounce off the wall
                if ((rl_pair_buf[0].s_x == 1)                                             
                        && (rl_pair_buf[1].s_x > 1)                                       
                        && (rl_pair_buf[0].s_v <= 3 && rl_pair_buf[1].s_v > 6)) {          
                                                                                          
                        r = -60;
                        return r;                
                }

                // 3.a) Fast right bounce, between the steps in none of the last two the ball was in the middle sector, cause the RL wasn't fast enough
                //      to capture it
                if (((rl_pair_buf[1].s_x < n_states_x) && (rl_pair_buf[1].s_x >= (s_x_middle + 4)))
                        && ((rl_pair_buf[1].s_x < n_states_x) && (rl_pair_buf[1].s_x >= (s_x_middle + 4)))
                        && (rl_pair_buf[1].s_v >= 4 && rl_pair_buf[1].s_v < 6)                                       
                        && (rl_pair_buf[0].s_v >= 8)) {

                        r = -60;
                        return r;       
                }

                // 3.a) Fast left bounce, between the steps in none of the last two the ball was in the middle sector, cause the RL wasn't fast enough
                //      to capture it

                if (((rl_pair_buf[1].s_x > 1) && (rl_pair_buf[1].s_x <= (s_x_middle - 4)))
                        && ((rl_pair_buf[1].s_x > 1) && (rl_pair_buf[1].s_x <= (s_x_middle - 4)))
                        && (rl_pair_buf[1].s_v > 9)                                                                 
                        && (rl_pair_buf[0].s_v >= 3)) {

                        r = -60;
                        return r;       
                }
                // OTHER NEGATIVE REWARDS :

                // 4.a) The ball in the last step was in the terminal sector with min speed and now is still in the terminal sector with minimum velocity:
                if (((rl_pair_buf[0].s_x == n_states_x && rl_pair_buf[1].s_x == n_states_x) ||  (rl_pair_buf[0].s_x == 1 && rl_pair_buf[1].s_x == 1))
                                && ((rl_pair_buf[1].s_v == 1 && rl_pair_buf[0].s_v == 1) || (rl_pair_buf[0].s_v == 6 && rl_pair_buf[1].s_v == 6))) {

                        r = -45;
                        return r;
                }

                
                // 4.b) Like 4.a) but the ball changed the speed direction meanwhile
                if (((rl_pair_buf[0].s_x == n_states_x && rl_pair_buf[1].s_x == n_states_x) ||  (rl_pair_buf[0].s_x == 1 && rl_pair_buf[1].s_x == 1))
                                && ((rl_pair_buf[1].s_v == 1 && (rl_pair_buf[0].s_v == 1 || rl_pair_buf[0].s_v == 6)) 
                                || (rl_pair_buf[1].s_v == 6 && (rl_pair_buf[0].s_v == 6 || rl_pair_buf[0].s_v == 1)))) {
                        
                        r = -45;
                        return r;
                }
                

                // 5.a) The ball moved toward the right beam limit between steps and didn't decreased his velocity
                //      the ball previous position was >= middle sector
                if (((rl_pair_buf[1].s_x >= s_x_middle) && (rl_pair_buf[0].s_x > rl_pair_buf[1].s_x)) 
                                && ((rl_pair_buf[0].s_v <= 5) && (rl_pair_buf[1].s_v <= 5))
                                && (rl_pair_buf[0].s_v >= rl_pair_buf[1].s_v)) {

                        r = -25;
                        return r;
                }

                // 5.b) The ball moved toward the left beam limit between steps and didn't decreased his velocity
                //      the ball previous position was >= middle sector
                if (((rl_pair_buf[1].s_x <= s_x_middle) && (rl_pair_buf[0].s_x < rl_pair_buf[1].s_x)) 
                                && ((rl_pair_buf[0].s_v > 5) && (rl_pair_buf[1].s_v > 5))
                                && (rl_pair_buf[0].s_v >= rl_pair_buf[1].s_v)) {

                        r = -25;
                        return r;
                }

                // Same as 5.a) but this time the ball didn't increased his sector, however it increased his velocity toward the right :
                if (((rl_pair_buf[1].s_x >= s_x_middle) && (rl_pair_buf[0].s_x >= rl_pair_buf[1].s_x)) 
                                && ((rl_pair_buf[0].s_v <= 5) && (rl_pair_buf[1].s_v <= 5))
                                && (rl_pair_buf[0].s_v > rl_pair_buf[1].s_v)) {

                        r = -25;
                        return r;
                }

                // Same as 5.a) but this time the ball didn't increased his sector, however it increased his velocity toward the left :
                if (((rl_pair_buf[1].s_x <= s_x_middle) && (rl_pair_buf[0].s_x <= rl_pair_buf[1].s_x)) 
                                && ((rl_pair_buf[0].s_v > 5) && (rl_pair_buf[1].s_v > 5 ))
                                && (rl_pair_buf[0].s_v > rl_pair_buf[1].s_v)) {

                        r = -25;
                        return r;
                }

                // 6.a) The ball was in the right part and one step further is in the left part with high velocity
                if ((rl_pair_buf[1].s_x > s_x_middle) && (rl_pair_buf[0].s_x < s_x_middle) && (rl_pair_buf[1].s_v > 5 || rl_pair_buf[1].s_v <= 1) 
                        && (rl_pair_buf[0].s_v > 8)   
                        && rl_pair_buf[1].s_v < 10) {

                        r = -20;
                        return r;        
                }

                // 6.b) The ball was in the left part and one step further is in the right part with high velocity
                if ((rl_pair_buf[1].s_x < s_x_middle) && (rl_pair_buf[0].s_x > s_x_middle) && (rl_pair_buf[1].s_v <= 5 || rl_pair_buf[1].s_v == 6) 
                        && (rl_pair_buf[0].s_v > 3 && rl_pair_buf[0].s_v <= 5)   
                        && rl_pair_buf[1].s_v < 5) {

                        r = -20;
                        return r;        
                }

                // 7.a) The ball was in the right part with velocity toward left and one step further is still in the same sector with velocity toward right
                //      action is not penalized if really close to the center
                if (rl_pair_buf[1].s_x >= (s_x_middle + 2) && (rl_pair_buf[1].s_x < (n_states_x)) && (rl_pair_buf[0].s_x >= rl_pair_buf[1].s_x)
                        && (rl_pair_buf[1].s_v > 5 ) 
                        && (rl_pair_buf[0].s_v >= 2)) {

                        r = -25;
                        return r;        
                }

                // 7.b) The ball was in the left part with velocity toward right and one step further is still in the same sector with velocity toward left
                //      action is not penalized if really close to the center
                if (rl_pair_buf[1].s_x <= (s_x_middle - 2) && (rl_pair_buf[1].s_x > 1) && (rl_pair_buf[0].s_x <= rl_pair_buf[1].s_x)
                        && (rl_pair_buf[1].s_v >= 1) // || (rl_pair_buf[1].s_v <= 7 && rl_pair_buf[1].s_v > 5))
                        && (rl_pair_buf[0].s_v > 6)) {

                        r = -25;
                        return r;        
                }

                // 8.a) The ball two steps before was near the right limit (with little velocity) and now is near the center with maximum velocity:
                if ((rl_pair_buf[2].s_x >= (n_states_x - 3)) && (rl_pair_buf[2].s_v <= 2 || (rl_pair_buf[2].s_v<=7 && rl_pair_buf[2].s_v>5)) 
                        && (rl_pair_buf[0].s_x <= (s_x_middle + 1)) && ((rl_pair_buf[0].s_v == 10))) {
                        
                        r = -20;
                        return r;
                }

                // 8.b) The ball two steps before was near the left limit (with little velocity) and now is near the center with maximum velocity:
                if ((rl_pair_buf[2].s_x <= 4) && (rl_pair_buf[2].s_v <= 2 || (rl_pair_buf[2].s_v<=7 && rl_pair_buf[2].s_v>5)) 
                        && (rl_pair_buf[0].s_x >= (s_x_middle - 1)) && ((rl_pair_buf[0].s_v == 5))) {
                        
                        r = -20;
                        return r;
                }
                
                // 9.a) Same as 8.a) but with no condition on the initial ball speed
                if ((rl_pair_buf[2].s_x >= (n_states_x - 4)) && (rl_pair_buf[0].s_x <= (s_x_middle + 1)) && ((rl_pair_buf[0].s_v == 10))) {
                        
                        r = -20;
                        return r;
                }

                // 9.b) Same as 8.b) but with no condition on the initial ball speed
                if((rl_pair_buf[2].s_x < 5) && (rl_pair_buf[0].s_x <= (s_x_middle - 1)) && ((rl_pair_buf[0].s_v == 5))){
                        
                        r = -20;
                        return r;
                }

                // 10) Give a penalty if the ball stais for 4 consecutive steps in the same sector with minimum velocity, but the sector isn't the middle one!
                if( ((rl_pair_buf[3].s_x == rl_pair_buf[2].s_x) && (rl_pair_buf[2].s_x == rl_pair_buf[1].s_x) && (rl_pair_buf[1].s_x == rl_pair_buf[0].s_x) && (rl_pair_buf[0].s_x != s_x_middle))
                                && ((rl_pair_buf[3].s_v == rl_pair_buf[2].s_v) && (rl_pair_buf[2].s_v == rl_pair_buf[1].s_v) && (rl_pair_buf[1].s_v == rl_pair_buf[0].s_v))){
                        
                        r = -50;
                        return r;
                }

                else    return 0;
}
//------------------------------------------------------------------------------------------------------------

void rl_buffer_init()
{
        rl_cbuffer.top = -1;    
        rl_cbuffer.n = 0;
}
//------------------------------------------------------------------------------------------------------------

void rl_buffer_insert(int s)
{
        rl_cbuffer.top = (rl_cbuffer.top + 1)%RL_BUF_DIM;
        rl_cbuffer.array[rl_cbuffer.top] = s;
        if (rl_cbuffer.n < RL_BUF_DIM) rl_cbuffer.n++;
}
//------------------------------------------------------------------------------------------------------------

int rl_buffer_get(int k)
{
int     j;
        if (rl_cbuffer.n == 0) return 0; 
        if (k > rl_cbuffer.n) k = rl_cbuffer.n;
        j = (rl_cbuffer.top - k + RL_BUF_DIM)% RL_BUF_DIM;
        return rl_cbuffer.array[j];
}
//------------------------------------------------------------------------------------------------------------

void set_rl_mode(int mode)
{       
        rl_mode = mode;
}
//------------------------------------------------------------------------------------------------------------
int get_rl_mode()
{
        return rl_mode;
}
//------------------------------------------------------------------------------------------------------------

void save_Q_matrix()
{
int             i, j;                           // Iterators        
const char*     file_name = "Q_matrix.txt";
FILE*           file = fopen(file_name, "w");

                if (!file) {
                printf("Can't open the file for saving the Q-matrix!\n");
                return;
                }

                for (i = 0; i < MAXSTA; i++) {
                        for (j = 0; j < MAXACT; j++) {
                                fprintf(file, "%d ", Q[i][j]);
                        }
                        fprintf(file, "\n");
                }

                fclose(file);
                printf("Q-matrix saved to file: %s\n", file_name);
}
//------------------------------------------------------------------------------------------------------------

void load_Q_matrix()
{
int             i, j;                           // Iterators        
const char*     file_name = "Q_matrix.txt";       
FILE*           file = fopen(file_name, "r");

                if (!file) {
                printf("Can't open the file for reading the Q-matrix!\n");
                return;
                }

                for (i = 0; i < MAXSTA; i++) {
                        for (j = 0; j < MAXACT; j++) {
                                fscanf(file, "%d", &Q[i][j]);
                        }
                }

                fclose(file);
                printf("Q-matrix correctly loaded from file: %s\n", file_name);
}
//------------------------------------------------------------------------------------------------------------

void set_rl_load_flag(int k)
{
        rl_load_flag = k;
}
//------------------------------------------------------------------------------------------------------------

void set_rl_save_flag(int k)
{
        rl_save_flag = k;
}
//------------------------------------------------------------------------------------------------------------

int get_rl_load_flag()
{
        return rl_load_flag;
}
//------------------------------------------------------------------------------------------------------------

int generate_random_position()
{
float   x_rand, v_rand;
        x_rand = frand(BALL_RADIUS, BEAM_LENGTH - BALL_RADIUS);          // random position on the beam
        v_rand = frand(-0.7, 0.7);                                       // random velocity in [-0.7, 0.7] m/s     

        // Setting the system status: (to the dynamics task)
        set_system_status(x_rand, v_rand);

        return state_2_state_rl(x_rand, v_rand);
}
//------------------------------------------------------------------------------------------------------------

int ball_stall_condition()
{
int             i;
int             counter = 0;
int             n_states, n_states_x, n_states_v;       // discretized states
int             s_x_middle;                             // s_x value at the beam center
rl_state_pair   rl_pair_buf[RL_BUF_DIM];                // array of discretized rl pairs (s_x, s_v)

                n_states = get_n_states();
                n_states_v = floor(sqrt((float)2/3*n_states));
                n_states_x = (float)n_states/n_states_v;

                s_x_middle = round((float)n_states_x/2);                              

                for (i = 0; i < RL_BUF_DIM; i++) {

                        rl_pair_buf[i] = get_rl_state_pair(rl_buffer_get(i));
                }

                // Check the stall condition:
                for (i = 0; i < RL_BUF_DIM; i++) {
                        if (rl_pair_buf[i].s_x == s_x_middle && (rl_pair_buf[i].s_v == 1)) counter++;
                }

                if (counter == RL_BUF_DIM) { 
                        stall_condition_flag = 1;
                        return 1;
                }        
                else return 0;
}
//------------------------------------------------------------------------------------------------------------

void set_simulation_speed(int i)
{
        simulation_speed = i;
}
//------------------------------------------------------------------------------------------------------------

int get_simulation_speed()
{
        return simulation_speed;
}
//------------------------------------------------------------------------------------------------------------

void set_ql_policy(int k)
{
        ql_policy = k;
}
//------------------------------------------------------------------------------------------------------------

int get_ql_policy()
{
        return ql_policy;
}
//------------------------------------------------------------------------------------------------------------

void update_beta_boltz(int episode)
{
float   beta_increment;                 // increment for Beta coefficient every episode
        beta_increment = 0.0015;        

        // Beta Coefficient Update:
        beta_boltz = episode*beta_increment;
}
//------------------------------------------------------------------------------------------------------------

float get_beta_boltz()
{
        return beta_boltz;
}
//------------------------------------------------------------------------------------------------------------
