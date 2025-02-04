#define _POSIX_C_SOURCE 200112L
//============================================================================================================
// BALL AND BEAM DYNAMICS HEADER
//============================================================================================================

//------------------------------------------------------------------------------------------------------------
// DYNAMICS TASK CONSTANTS
//------------------------------------------------------------------------------------------------------------
#define DYNAMICS_PRIO 90    // task priority
#define DYNAMICS_DL   30    // relative deadline in [ms]
#define DYNAMICS_PER  30    // period in [ms]

//------------------------------------------------------------------------------------------------------------
// BALL AND BEAM CONSTANTS
//------------------------------------------------------------------------------------------------------------
#define BALL_MASS   0.110               // mass of the ball [kg], Hp. ball of steel (7.8 kg/dm^3)
#define BALL_RADIUS 0.015               // radius of the ball [m]
#define G0          9.81                // gravity acceleration [m/s^2]
#define BEAM_LENGTH 0.75                // length of the beam [m]
#define T_DYNAMICS  0.030               // Delta-T of the discretized dynamics [s], = 30 [ms]
#define MOTOR_TAU   0.35                // Motor Time-constant [s]
#define NUM_ACTIONS 7                   // Number of discretized actions (Odd number!)
#define THETA_STEP  5                   // Minimum theta step [degrees]
#define ROLLING_FRICTION_COEFF 0.05     // Rolling friction coefficient
#define DISTURBING_VELOCITY 0.45        // Additional velocity value after a disturbing action [m/s]

// Disturbance enum:
enum {LEFT_PUSH = -1, NO_PUSH, RIGHT_PUSH};  

// Discretized RL State Pair:
typedef struct {
    int s_x;    // from 1 to (n_states_x)
    int s_v;    // from 1 to (n_states_v)
} rl_state_pair;

//------------------------------------------------------------------------------------------------------------
// BALL AND BEAM STATE
//------------------------------------------------------------------------------------------------------------
struct status {
    float   x;          // relative ball position on the beam [m]
    float   v;          // relative speed of the ball [m/s]
    float   a;          // relative acceleration of the ball [m/s^2]
    float   theta;      // angle of the beam [rad]
    float   omega;      // angular speed of the beam [rad/s]
    float   alpha;      // angular acceleration of the beam [rad/s^2]
    float   theta_ref;  // Input to the servo-motor [rad] 
    int     s;          // System discretized state (it is a combination of s_x and s_v discretized states)
    int     action;     // System discretized input/action
};

//------------------------------------------------------------------------------------------------------------
// SYSTEM FUNCTIONS PROTOTYPES:
//------------------------------------------------------------------------------------------------------------

/* 
This is the transition function, it computes the next system state if the simulation speed is set to "NORMAL".
If the simulation speed is instead "FAST" it will compute the system state after 15 dynamics steps before waiting
for the next task period.
*/
void ball_and_beam_dynamics();

/*
This function encodes the discretized input angles [rad] into (int) actions
*/
int theta_ref2action(float theta_ref);

/*
This function decodes the action (int) into motor angles values [rad]
*/
float action2theta_ref(int action);

/*
System dynamics task function
*/
void* system_dynamics_task(void* arg);

/*
This function initialize the system state and creates the dynamics task 
*/
void system_dynamics_init();

/*
This function handles the beam limits: the ball will bounce off the limits when close enough
*/
void handle_beam_limits();

/*
Get a copy of the current status for other modules/libraries to see it
*/
struct status get_system_status();

/*
This function sets the system status based on (x, v) state pair, other variables like theta, omega etc. are set to 0
The function is used inside the RL loop to generate a random ball starting position
*/
void set_system_status(float x, float v);

/*
This function set the desired theta_ref angle
*/
void set_theta_ref(float theta_ref);

/*
End the dynamics and all the tasks setting the end flag to 1 in the system status
*/
void end_simulation();

/*
This function check the end flag, if end = 0, it returns 0, 1 otherwise
*/
int is_end();

/*
This function output the RL discretized state pair (s_x and s_v) value given the state s
*/
rl_state_pair get_rl_state_pair(int s);

/*
This function outputs the discretized state s, given the current system's status (x,v) values
*/
int state_2_state_rl(float x, float v);

/*
This function sets the disturbing force flag to a value between {-1, 0, 1}
*/
void set_disturbing_force_flag(int k);

/*
This function change the rolling friction coefficient, current value + delta
*/
void change_friction_coeff(float delta_f);

/*
This function returns the current value of the friction coefficient
*/
float get_friction_coeff();
