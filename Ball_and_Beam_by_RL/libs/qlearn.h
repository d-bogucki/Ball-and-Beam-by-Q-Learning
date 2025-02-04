#define _POSIX_C_SOURCE 200112L
//============================================================================================================
// Q-LEARNING CUSTOM LIBRARY HEADER
//============================================================================================================

//------------------------------------------------------------------------------------------------------------
// GLOBAL Q-LEARN CONSTANTS:
//------------------------------------------------------------------------------------------------------------
#define MAXSTA 150      // max number of states
#define MAXACT 15       // max number of actions
#define ALPHA0 1.0      // Default learning rate
#define EPSINI 1.0      // initial exploration factor
#define EPSFIN 0.01     // final exploration factor
#define GAMMA0 0.9      // default discount factor
#define DECAY0 0.99     // default espilon decay rate

#define MAXSTEPS 2000   // Max number of steps inside an episode
#define MAXEPI   350    // Max number of episodes

#define RL_BUF_DIM 5    // Max number of previous states saved in a circular buffer

//------------------------------------------------------------------------------------------------------------
// TASK CONSTANTS
//------------------------------------------------------------------------------------------------------------
#define RL_PER   450    // task period in ms 
#define RL_PER_FAST 30  // task period in ms, when in "FAST" simulation mode
#define RL_DL    450    // relative deadline in ms
#define RL_DL_FAST 30   // relative deadline in ms when in "FAST" mode
#define RL_PRI   90     // task priority

//------------------------------------------------------------------------------------------------------------
// Circular Buffer Struct:
typedef struct {
    int array[RL_BUF_DIM];
    int top;
    int n;
} circular_buffer; 

// Enum definitions:
enum {NORMAL, FAST};

enum {LEARNING_MODE, EXPLOIT_MODE, PLAY_MODE};

enum {EPS_GREEDY, BOLTZMANN};


//------------------------------------------------------------------------------------------------------------
// Q-LEARN FUNCTION PROTOTYPES
//------------------------------------------------------------------------------------------------------------

/*
This function initializes the number of states and actions, along with the learning rate, discount factor,
initial exploration probability, final exploration probability, decay rate and Q matrix value
*/
void ql_init(int ns, int na);

/* 
This function reduces the current exploration probability
*/
void ql_reduce_exploration();

/*
This function returns the maximum value Q[s][a] for all possible actions, given a certain state "s"
*/
float ql_maxQ(int s);

/* 
This function implements the Boltzmann policy for action selection. It returns the action according to the policy, given a state "s"
*/
int ql_boltzmann_policy(int s);

/* 
This function returns the action with the maximum Q value within a given state s
*/
float ql_best_action(int s);

/*
This function returns the action according to the eps-greedy policy, given a state "s"
*/
int ql_egreedy_policy(int s);

/* 
This function updates element Q[s][a] of the Q-matrix, based on the reward and the new state
*/
float ql_updateQ(int s, int a, int r, int snew);

/*
This function performs a learning cycle within an episode:
*/
float learn_episode(int s0, int i, int episode_counter);

/*
This function implements a full learning loop:
*/
void qlearn(float eps, int i);

/*
RL task function 
*/
void* RL_task(void* arg);

/*
RL init function and task creation
*/
void RL_init();

/*
This function set the current learning rate to a desired value
*/
void ql_set_learning_rate(float lr);

/*
This function set the current discount factor to a desired value
*/
void ql_set_discount_factor(float df);

/*
This function set the current exploration range to a desired value
*/
void ql_set_expl_range(float eini, float efin);

/*
This function set the current exploration decay to a desired value
*/
void ql_set_expl_decay(float d);

/*
This function returns the current learning rate 
*/
float ql_get_learning_rate();

/*
This function returns the current discount factor
*/
float ql_get_discount_factor();

/*
This function returns the current exploration decay
*/
float ql_get_expl_decay();

/*
This function returns the current epsilon value
*/
float ql_get_epsilon();

/*
This function returns a random float number between xmi and xma
*/
float frand(float xmi, float xma); 

// ---------------------------------------------------------------------------
// Other custom functions:

/*
This function returns the current value of number of discretized states
*/
int get_n_states();

/*
This function returns the current number of discretized actions
*/
int get_n_actions();

/*
This function returns the reward, given the current state and the (RL_BUF_DIM-1) previous states
*/
int get_reward();

/*
This function gets the value of the reward of the Q matrix at specified position in the matrix (state, action)
*/
int get_Q_value(int state, int action);

/*
This function sets the rl_mode static variable to a specific value
*/
void set_rl_mode(int mode);

/*
This function returns the current RL mode
*/
int get_rl_mode();

/*
This function saves the content of the Q-matrix inside a .txt file
*/
void save_Q_matrix();

/*
This function loads the content of the Q-matrix from a .txt file
*/
void load_Q_matrix();

/*
This function sets the rl_load_flag, acceptable values = {0,1}
*/
void set_rl_load_flag(int k);

/*
This function sets the rl_save_flag, acceptable values = {0,1}
*/
void set_rl_save_flag(int k);

/*
This function returns the current rl_load_flag
*/
int get_rl_load_flag();

/*
This function generates a random starting position for the ball at the start of a new episode, returns the discretized state s
*/
int generate_random_position();

/*
This function check if the ball stays at the center for RL_BUF_DIM number of steps, returns 1 if true, 0 otherwise
*/
int ball_stall_condition();

/*
This function sets the simulation speed, in particular it sets the RL task period, also the dynamic task will speed up the dynamics accordingly
*/
void set_simulation_speed(int i);

/*
This function returns the current simulation speed. It can be either "NORMAL" (0) or "FAST" (1)
*/
int get_simulation_speed();

/*
This function initializes the circular buffer
*/
void rl_buffer_init();

/*
This function adds a new state inside the circular buffer
*/
void rl_buffer_insert(int s);

/*
This function returns the k-th element inside the circular buffer
*/
int rl_buffer_get(int k);

/*
This function sets the Q-Learning Policy, either "EPS_GREEDY" or "BOLTZMANN"
*/
void set_ql_policy(int k);

/*
This function returns the current Q-Learning Policy
*/
int get_ql_policy();

/*
This function updates the Beta coefficient for the Boltzmann softmax distribution to 
a specific value, related to the function chosen for Beta and the episode number 
*/
void update_beta_boltz(int episode);

/*
This function returns the current beta boltzmann coefficient
*/
float get_beta_boltz();
