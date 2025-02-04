#define _POSIX_C_SOURCE 200112L
//============================================================================================================
// GRAPHICS LIBRARY HEADER FOR BALL AND BEAM SYSTEM
//============================================================================================================

//------------------------------------------------------------------------------------------------------------
// GRAPHICS TASK CONSTANTS
//------------------------------------------------------------------------------------------------------------
#define GRAPHICS_PRIO   80    // task priority
#define GRAPHICS_DL     80    // relative deadline in [ms]
#define GRAPHICS_PER    30    // period in [ms]  

//------------------------------------------------------------------------------------------------------------
// GRAPHICS CONSTANTS
//------------------------------------------------------------------------------------------------------------
// Graphics window resolution in pixels:
#define WIN_X 1024
#define WIN_Y 768
// Ball and Beam visualization box size in pixels:
#define SYSTEM_BOX_WIDTH 640
// Distance of boxes from window corners in pixels:
#define BORDER_DIST 5
// Ball and Beam visualization box corners position in pixels:
#define UPPER_LEFT_SYS_BOX_X (BORDER_DIST)
#define UPPER_LEFT_SYS_BOX_Y 273
#define LOWER_RIGHT_SYS_BOX_X (BORDER_DIST + SYSTEM_BOX_WIDTH)
#define LOWER_RIGHT_SYS_BOX_Y (WIN_Y - BORDER_DIST)
#define SYSTEM_BOX_HEIGHT (LOWER_RIGHT_SYS_BOX_Y - UPPER_LEFT_SYS_BOX_Y)
// Menu Area visualization box corners position in pixels:
#define UPPER_LEFT_MENU_BOX_X (BORDER_DIST)
#define UPPER_LEFT_MENU_BOX_Y (BORDER_DIST)
#define LOWER_RIGHT_MENU_BOX_X (LOWER_RIGHT_SYS_BOX_X)
#define LOWER_RIGHT_MENU_BOX_Y (UPPER_LEFT_SYS_BOX_Y - BORDER_DIST)
// Menu Area separator x position: (relative to left menu edge)
#define MENU_BOX_SEPARATOR 300
// Q-Matrix area:
#define UPPER_LEFT_Q_BOX_X (LOWER_RIGHT_MENU_BOX_X + BORDER_DIST)
#define UPPER_LEFT_Q_BOX_Y (UPPER_LEFT_MENU_BOX_Y)
#define LOWER_RIGHT_Q_BOX_X (WIN_X - BORDER_DIST)
#define LOWER_RIGHT_Q_BOX_Y (WIN_Y - BORDER_DIST)
// Window Background color: (8-BIT VGA PALETTE)
#define WIN_BKG 0
#define SYSTEM_BOX_BKG 15
#define MENU_BOX_COL 7
#define Q_MATRIX_BKG 85
// Q-Table position:
#define Q_TABLE_START_X 860     // Upper left corner x position in pixels of Q-table
#define Q_TABLE_START_Y 20      // Upper left corner y position in pixels of Q-table
#define CELL_SIZE 8             // Q-table cell size 
// Menu Theta Plot:
#define PLOT_WIDTH 320
#define PLOT_HEIGHT 110
#define UPPER_LEFT_PLOT_X (BORDER_DIST + MENU_BOX_SEPARATOR + 10)
#define UPPER_LEFT_PLOT_Y (UPPER_LEFT_MENU_BOX_Y + 148)
#define LOWER_RIGHT_PLOT_X (UPPER_LEFT_PLOT_X + PLOT_WIDTH)
#define LOWER_RIGHT_PLOT_Y (UPPER_LEFT_PLOT_Y + PLOT_HEIGHT)
#define PLOT_BKG 32
// Status box:
#define UPPER_LEFT_STATUS_X ( UPPER_LEFT_PLOT_X - 1)
#define UPPER_LEFT_STATUS_Y (UPPER_LEFT_MENU_BOX_Y + 1)
#define LOWER_RIGHT_STATUS_X (LOWER_RIGHT_PLOT_X + 1)
#define LOWER_RIGHT_STATUS_Y (UPPER_LEFT_PLOT_Y  - 40 + 2)
#define RL_STATUS_BOX_WIDTH (LOWER_RIGHT_STATUS_X - UPPER_LEFT_STATUS_X)
#define RL_STATUS_BOX_HEIGHT (LOWER_RIGHT_STATUS_Y - UPPER_LEFT_STATUS_Y)
// TD-Error Plot Box:
#define TD_ERR_PLOT_HEIGHT 120
#define UPPER_LEFT_TD_ERR_PLOT_X (UPPER_LEFT_Q_BOX_X + 2*BORDER_DIST)
#define UPPER_LEFT_TD_ERR_PLOT_Y (LOWER_RIGHT_Q_BOX_Y - TD_ERR_PLOT_HEIGHT - 2*BORDER_DIST)
#define LOWER_RIGHT_TD_ERR_PLOT_X (LOWER_RIGHT_Q_BOX_X - 2*BORDER_DIST)
#define LOWER_RIGHT_TD_ERR_PLOT_Y (UPPER_LEFT_TD_ERR_PLOT_Y + TD_ERR_PLOT_HEIGHT)
#define TD_ERR_PLOT_WIDTH (LOWER_RIGHT_TD_ERR_PLOT_X - UPPER_LEFT_TD_ERR_PLOT_X)
// Menu Arrow Constants:
#define KEY_SIZE 30
#define KEY_L1 5
#define KEY_L2 3
#define KEY_L3 5
#define KEY_D2 60
#define KEY_D1 50
#define ARROW_THICKNESS 2
#define TRIANGLE_THICKNESS 12
#define KEY_TEXT_WIDTH 90
#define KEY_TEXT_HEIGHT 12
// Smaller button constants:
#define KEY_SIZE_2 16
#define KEY_L1_2 3
#define KEY_L2_2 2
#define KEY_L3_2 3

// BALL AND BEAM SYSTEM COMPONENTS DIMENSIONS:
#define L1 0.1                              // distance of left base edge along x direction from World reference system [m]
#define BASE_WIDTH 0.3                      // [m]
#define BASE_HEIGHT 0.05                    // [m]
#define PIVOT_X (L1 + BASE_WIDTH/2)         // x coordinate of beam hinge in world coordinates [m]
#define PIVOT_Y 0.45                        // y coordinate of beam hinge in world coordinates [m]
#define PIVOT_DISTANCE 0.05                 // Hinge distance from beam left limit [m]
#define BEAM_THICKNESS 0.02                 // [m]
#define BEAM_THICKNESS_2 0.02               // thickness of the beam limits [m]
#define BEAM_HEIGHT 0.06                    // [m]
#define SUPPORT_THICKNESS 0.045             // [m]
// Motor Link:
#define LINK_HEIGHT 0.04                    // Motor link height [m]
#define LINK_LENGTH 0.25                    // Motor link length [m]
#define LINK_D1 (LINK_HEIGHT/4)             // Distance of the hinge from end of the link [m]
#define MOTOR_CENTER_X (PIVOT_X - PIVOT_DISTANCE + 0.75 - LINK_LENGTH) // DC Motor center x position [m]
#define MOTOR_CENTER_Y 0.20                 // DC Motor center y position [m]
// Motor Chassis:
#define MOTOR_SIZE 0.15                     // [m]
#define MOTOR_COL 23
// Laser:
#define LASER_L1 (BEAM_THICKNESS_2)         // [m]
#define LASER_L2 (BEAM_THICKNESS_2/2)       // [m]


// Forward definitions: (to avoid including other libraries in this header)
struct BITMAP;
struct status;

//------------------------------------------------------------------------------------------------------------
// GRAPHICS FUNCTION PROTOTYPES
//------------------------------------------------------------------------------------------------------------

/*
Initialize all the allegro functionalities, draws all the static graphics parts and creates the graphics task
*/
void graphics_init();

/*
Update all the graphics. The 4 input are the bitmaps of the beam, the motor link and the two bitmap for the system box and the RL status box
*/
void update_graphics(struct BITMAP* system_bitmap, struct BITMAP* beam_bitmap, struct BITMAP* link_bitmap, struct BITMAP* rl_status_bitmap);

/*
Updates the system box graphics
*/
void update_system_graphics(struct status current_status, struct BITMAP* system_bitmap, struct BITMAP* beam_bitmap, struct BITMAP* link_bitmap);

/*
Draw graphics boxes (system box, menu and status box)
*/
void draw_boxes();

/*
Graphics Task function
*/
void* graphics_task(void* arg);

/*
This function converts meters value to the current pixel representation
*/
int meters2pixels(float x);

/*
This function converts world frame x coordinate to x pixel coordinate
*/
int world2pixel_x_coordinate(float x);

/*
This function converts world frame y coordinate to y pixel coordinate
*/
int world2pixel_y_coordinate(float y);

/*
This function draws the initial empty Q-table matrix
*/
void draw_Q_table();

/*
This function sets all the necessary RL state values that are displayed in the graphics
*/
void set_graphics_RL_state(int state, int action, int reward, int step, int episode_counter, float average_TD_error);

/*
This function updates the Q-table graphics:
*/
void update_Q_table_graphics();

/*
Draw Q-Table after loading the Q-matrix from a .txt file
*/
void draw_Q_table_after_loading();

/*
This function draws the menu arrows:
*/
void draw_menu_arrows();

/*
This function clears ONLY the bitmaps and boxes that need to be re-drawn every period 
*/
void clear_boxes(struct BITMAP* rl_status_bitmap, struct BITMAP* beam_bitmap, struct BITMAP* system_bitmap);

/*
This function draws the RL modes text in the static menu area
*/
void draw_RL_modes_text();

/*
This function converts the (float) theta angle of the beam into pixel coordinates for plotting
*/
int theta_2_plot_pixels(float theta);

/*
This function updates the theta plot, based on the current system state
*/
void update_theta_plot(struct status current_status);

/*
This function draws the DC motor chassis
*/
void draw_motor_chassis(struct BITMAP* system_bitmap);

/*
This function draws the laser sensor. The inputs are the beam bitmap and the ball x position
*/
void draw_laser(struct BITMAP* beam_bitmap, float x);

/*
This function updates the RL status text in the menu box
*/
void draw_RL_status(struct BITMAP* rl_status_bitmap);

/*
This function draws a button key, A_x, A_y are the position of a specific corner, "text" input is the text to be written on the button
*/
void draw_button_key(int A_x, int A_y, char* text);

/*
This function draws the live value of ball relative status (x,v) and friction coefficient inside the system box 
*/
void draw_ball_status(struct BITMAP* system_bitmap, struct status system_state);

/*
This function sets the flag for drawing the loaded Q-table matrix
*/
void set_q_table_graphics_flag(int i);

/*
This function draws the motor link inside the given input bitmap
*/
void draw_link_bitmap(struct BITMAP* link_bitmap);

/*
This function draws the beam bitmap in the input bitmap
*/
void draw_beam_bitmap(struct BITMAP* beam_bitmap);

/*
This function draws the beam main support
*/
void draw_beam_support(struct BITMAP* beam_bitmap);

/*
This function draws the second link of the system 4R structure, based on theta angle of the beam and motor angle
*/
void draw_link_2(float theta_beam, float theta_motor, struct BITMAP* system_bitmap);

/*
This function draws the part where the second link connects to the beam, based on the current theta value of the beam
*/
void draw_beam_support_2(float theta_beam, struct BITMAP* system_bitmap);

/*
This function draws the legend for the Q-table coloring
*/
void draw_Q_table_legend();

/*
This function draws the Q-table graphic instructions when pressing specific keyboard buttons
*/
void draw_Q_table_button_instructions();

/*
This function draws the empty TD-error plot
*/
void draw_empty_TD_err_plot();

/*
This function converts the average TD error value over an episode to its y representation on the screen inside the TD-Error plot:
*/
int td_err_2_y_plot(float td_error);

/*
This function updates the TD-Error plot
*/
void update_TD_err_plot();

/*
This function sets the average TD error over an episode to a given value
*/
void set_avg_episode_TD_err(float td_err);
