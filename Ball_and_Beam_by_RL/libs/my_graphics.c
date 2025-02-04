#define _POSIX_C_SOURCE 200112L
//============================================================================================================
// GRAPHICS LIBRARY FOR BALL AND BEAM SYSTEM
//============================================================================================================

// Standard libraries:
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>

// Custom Libraries headers:
#include "ptask.h"
#include "my_graphics.h"
#include "qlearn.h"
#include <allegro.h>                    
#include "ball_and_beam_dynamics.h"     

// PI [rad] constant definition for math calculations:
#define PI 3.14159265358979323846

//------------------------------------------------------------------------------------------------------------
// GRAPHICS GLOBAL VARIABLES:
//------------------------------------------------------------------------------------------------------------

static int rl_step_flag = 0;                    // RL Step flag (it set to 1 by the RL task every time the latter has performed a step)
                                                // After updating the reward in the Q-table, it is set again to 0.
static int reward_state = 0;                    // Buffer for RL current reward state
static int reward_action = 0;                   // Buffer for RL current reward action
static int current_reward = 0;                  // Buffer for RL current reward value
static int rl_step = 0;                         // Buffer for the RL current step value
static int rl_episode_counter = 0;              // Buffer for the RL current episode counter
static float avg_TD_error = 0.0;                // Buffer for the RL current average TD error
static float avg_episode_TD_error = 0.0;        // Buffer for average TD over the last episode
// Old reward, reward state and reward action values:
static int reward_state_old = 0;
static int reward_action_old = 0;
static int reward_old = 0;
// Theta plot drawing variables:
static int plot_step = 0;                       // Theta plot step
static int plot_x_pixel_old = UPPER_LEFT_PLOT_X;
static int plot_y_pixel_old = UPPER_LEFT_PLOT_Y + PLOT_HEIGHT/2;
// Flag for drawing the loaded Q-Matrix:
static int q_table_load_graphics_flag = 0;
// TD-Error plot:
static int td_plot_flag = 1;
static int td_plot_step = 0;
static int td_plot_x_pixel_old;
static int td_plot_y_pixel_old;

// Default 8x8 allegro font:
extern FONT *font;                              

//------------------------------------------------------------------------------------------------------------
// GRAPHICS FUNCTIONS DEFINITIONS:
//------------------------------------------------------------------------------------------------------------

void graphics_init()
{
int     tret;                                   // task_create() return flag                         
        allegro_init();                         // Allegro initialization
        set_color_depth(8);                     // 8-bit color depth
        set_gfx_mode(GFX_AUTODETECT_WINDOWED, WIN_X, WIN_Y, 0, 0);
        clear_to_color(screen, WIN_BKG);
        
        draw_boxes();                           // Draws the system, menu, and Q-table boxes 
        draw_menu_arrows();                     // Draws the menu arrows
        draw_RL_modes_text();                   // Draws the RL modes instructions in the menu
        draw_Q_table();                         // Draws the empty Q-table 

        tret = task_create(2, graphics_task, GRAPHICS_PER, GRAPHICS_DL, GRAPHICS_PRIO, ACT); 
        if (tret == 0) {
                printf("Graphics task created successfully!\n");
        }
        else {
                printf("Graphic task not created successfully!\n");
        }    
}
//------------------------------------------------------------------------------------------------------------

void* graphics_task(void* arg)
{
int     i;  // task index
int     beam_bitmap_width, beam_bitmap_height;
int     link_bitmap_width, link_bitmap_height;

//------------------------------------------------------------------------
// BITMAPS CREATION (ONLY ONCE AT STARTUP)
//------------------------------------------------------------------------
        beam_bitmap_width = meters2pixels(BEAM_LENGTH) + meters2pixels(BEAM_THICKNESS_2*2) + meters2pixels(LASER_L1 + LASER_L2);
        beam_bitmap_height = meters2pixels(BEAM_HEIGHT);
        link_bitmap_width = meters2pixels(LINK_LENGTH);
        link_bitmap_height = meters2pixels(LINK_HEIGHT);
// Beam Bitmap:
BITMAP* beam_bitmap = create_bitmap(beam_bitmap_width, beam_bitmap_height);
        clear_bitmap(beam_bitmap); 
// Link Bitmap:        
BITMAP* link_bitmap = create_bitmap(link_bitmap_width, link_bitmap_height);
        clear_bitmap(link_bitmap);  
        // Motor Link Drawing:
        draw_link_bitmap(link_bitmap); 
// System Box Bitmap:
BITMAP* system_bitmap = create_bitmap(SYSTEM_BOX_WIDTH, SYSTEM_BOX_HEIGHT);
        clear_bitmap(system_bitmap);
// RL status Bitmap:
BITMAP* rl_status_bitmap = create_bitmap(RL_STATUS_BOX_WIDTH - 4, RL_STATUS_BOX_HEIGHT - 4);
        clear_to_color(rl_status_bitmap, makecol8(255,255,255)); 
//------------------------------------------------------------------------                      

        i = get_task_index(arg);
        wait_for_activation(i);        
        
        while (!is_end()) {
                   
                clear_boxes(rl_status_bitmap, beam_bitmap, system_bitmap);                      // Clear all the dynamic graphics at every loop, before drawing them again
                update_graphics(system_bitmap, beam_bitmap, link_bitmap, rl_status_bitmap);     // Update the graphics

                // Draw the loaded Q-matrix if the corresponding flag is set to 1:
                if (q_table_load_graphics_flag) {
                        draw_Q_table_after_loading();
                        // Then put the flag to 0, otherwise it will be drawn at every cicle
                        q_table_load_graphics_flag = 0;
                }  
                
                // Deadline miss check:
                if (deadline_miss(i)) {
                        printf("The Graphics Task missed the deadline!\n");
                }

                wait_for_period(i);             
        }
        return NULL;
}
//------------------------------------------------------------------------------------------------------------

void draw_boxes()
{       
        // System Box Area:
        rectfill(screen, UPPER_LEFT_SYS_BOX_X, UPPER_LEFT_SYS_BOX_Y ,
                        LOWER_RIGHT_SYS_BOX_X, LOWER_RIGHT_SYS_BOX_Y,
                        SYSTEM_BOX_BKG);
        // Menu Box Area:
        rectfill(screen, UPPER_LEFT_MENU_BOX_X, UPPER_LEFT_MENU_BOX_Y,
                        LOWER_RIGHT_MENU_BOX_X, LOWER_RIGHT_MENU_BOX_Y,
                        MENU_BOX_COL);
        // Q-Table Box Area:
        rectfill(screen, UPPER_LEFT_Q_BOX_X, UPPER_LEFT_Q_BOX_Y,
                        LOWER_RIGHT_Q_BOX_X, LOWER_RIGHT_Q_BOX_Y,
                        Q_MATRIX_BKG);
        // Theta-Plot Borders:
        rectfill(screen, UPPER_LEFT_PLOT_X-1, UPPER_LEFT_PLOT_Y-1,
                        LOWER_RIGHT_PLOT_X+1, LOWER_RIGHT_PLOT_Y+1,
                        46); 
        // Theta-Plot Area:
        rectfill(screen, UPPER_LEFT_PLOT_X, UPPER_LEFT_PLOT_Y,
                        LOWER_RIGHT_PLOT_X, LOWER_RIGHT_PLOT_Y,
                        PLOT_BKG);

        // Theta-Plot Title:
        rectfill(screen, UPPER_LEFT_PLOT_X, UPPER_LEFT_PLOT_Y - 17, UPPER_LEFT_PLOT_X + 95, UPPER_LEFT_PLOT_Y - 5, 16);
        rectfill(screen, UPPER_LEFT_PLOT_X + 2, UPPER_LEFT_PLOT_Y - 17 + 2, UPPER_LEFT_PLOT_X + 95 + 2, UPPER_LEFT_PLOT_Y - 5 + 2, 16);
        rectfill(screen, UPPER_LEFT_PLOT_X + 1, UPPER_LEFT_PLOT_Y - 17 + 1, UPPER_LEFT_PLOT_X + 95 -1, UPPER_LEFT_PLOT_Y - 5 - 1, 15);
        textout_ex(screen, font, "THETA PLOT:", UPPER_LEFT_PLOT_X + 1 + 2, UPPER_LEFT_PLOT_Y - 17 + 1 + 2, 16, -1);

        // Theta-Plot Zero Green Line:
        line(screen, UPPER_LEFT_PLOT_X, UPPER_LEFT_PLOT_Y + PLOT_HEIGHT/2, UPPER_LEFT_PLOT_X + PLOT_WIDTH, UPPER_LEFT_PLOT_Y + PLOT_HEIGHT/2, 46); 
                      
        // RL Status Area:
        rectfill(screen, UPPER_LEFT_STATUS_X, UPPER_LEFT_STATUS_Y, LOWER_RIGHT_STATUS_X, LOWER_RIGHT_STATUS_Y, 16);
        rectfill(screen, UPPER_LEFT_STATUS_X + 2, UPPER_LEFT_STATUS_Y + 2, LOWER_RIGHT_STATUS_X - 2, LOWER_RIGHT_STATUS_Y - 2, 15);

        // TD-Error Plot Box:
        rectfill(screen, UPPER_LEFT_TD_ERR_PLOT_X - 1, UPPER_LEFT_TD_ERR_PLOT_Y - 1, LOWER_RIGHT_TD_ERR_PLOT_X + 1, LOWER_RIGHT_TD_ERR_PLOT_Y + 1, 16);
        rectfill(screen, UPPER_LEFT_TD_ERR_PLOT_X, UPPER_LEFT_TD_ERR_PLOT_Y, LOWER_RIGHT_TD_ERR_PLOT_X, LOWER_RIGHT_TD_ERR_PLOT_Y, makecol8(255,255,255));
        draw_empty_TD_err_plot();

}
//------------------------------------------------------------------------------------------------------------

void clear_boxes(BITMAP* rl_status_bitmap, BITMAP* beam_bitmap, BITMAP* system_bitmap)
{       
        // Clear the System bitmap:
        clear_to_color(system_bitmap, SYSTEM_BOX_BKG);

        // Clear Beam Bitmap:
        clear(beam_bitmap);

        // Clear the RL-State bitmap at every RL step:
        if (rl_step_flag) {
        clear_to_color(rl_status_bitmap, SYSTEM_BOX_BKG);
        } 

        // Clear the theta value at every step:        
        rectfill(screen, UPPER_LEFT_PLOT_X + 180 - 2,  UPPER_LEFT_PLOT_Y - text_height(font) - 3 - 2, LOWER_RIGHT_PLOT_X, UPPER_LEFT_PLOT_Y -2, MENU_BOX_COL);                   
}

//------------------------------------------------------------------------------------------------------------

void update_graphics(BITMAP* system_bitmap, BITMAP* beam_bitmap, BITMAP* link_bitmap, BITMAP* rl_status_bitmap)
{   
struct status   current_status;                 // Current System Status buffer
int             sim_speed;                      // Current Simulation Speed

                current_status = get_system_status();
                sim_speed = get_simulation_speed();

                // System graphics update:
                update_system_graphics(current_status, system_bitmap, beam_bitmap, link_bitmap);
                // Q-table graphics update:                 
                update_Q_table_graphics();
                // Draws the live theta plot only if the simulation speed is set to NORMAL:
                if (sim_speed == NORMAL) {
                update_theta_plot(current_status);
                } 
                // RL-status graphics update:
                draw_RL_status(rl_status_bitmap);
                // TD-Error Plot update:
                update_TD_err_plot();
                // After the drawings the RL step flag is set back to 0:  
                rl_step_flag = 0;
}
//------------------------------------------------------------------------------------------------------------

int meters2pixels(float x)
{
int     pixels_per_meter, result;
        pixels_per_meter = floor(SYSTEM_BOX_WIDTH/BEAM_LENGTH);         // around 853 pixels/meter
        pixels_per_meter = pixels_per_meter - 300;                      // subtracting some pixels not to occupy all the sys box (now 553 pixel/meter)
                                                    
        result = pixels_per_meter * x;
        return result;
}
//------------------------------------------------------------------------------------------------------------

int world2pixel_x_coordinate(float x)
{
int     pixel_x;
        pixel_x = meters2pixels(x);       
        return pixel_x;
}
//------------------------------------------------------------------------------------------------------------

int world2pixel_y_coordinate(float y)
{
int     pixels_per_meter, pixel_y;
        pixels_per_meter = floor(SYSTEM_BOX_WIDTH/BEAM_LENGTH);         // around 853 pixels/meter
        pixels_per_meter = pixels_per_meter - 300;                      // subtracting some pixels not to occupy all the sys box

        pixel_y = meters2pixels((float)SYSTEM_BOX_HEIGHT/pixels_per_meter - y);
        return pixel_y;

}
//------------------------------------------------------------------------------------------------------------

void update_system_graphics(struct status current_status, BITMAP* system_bitmap, BITMAP* beam_bitmap, BITMAP* link_bitmap)
{
int     sim_speed = get_simulation_speed(); 
int     border_size = 2;                                // Size in pixels of black contour of drawing components
int     border_color = 16;                              // Almost black
// BEAM VARIABLES:
int     pivot_x_beam, pivot_y_beam;                     // Beam hinge position in pixels inside the beam bitmap
float   scale = 1;                                      // scale factor
float   theta = current_status.theta * 180/PI + 90;     // angle in degrees [0,360]
float   angle;                                          // angle in [0,256] for Allegro functions
        angle = 64 - theta*(float)128/180;
float   theta_true;                                     // Theta true value (0 -> beam is horizontal)
        theta_true = (theta - 90)*PI/180;        
// BALL VARIABLES:
int     ball_color = 11;
float   x = current_status.x;                           // ball relative position [m] along the beam
float   ball_x, ball_y;                                 // ball absolute coordinates in world frame ([m], [m])
int     ball_x_pixel, ball_y_pixel;                     // pixel coordinates of the ball in the screen
// MOTOR LINK VARIABLES:   
int     pivot_x_link, pivot_y_link;                     // Link hinge position inside the link bitmap     
float   theta_motor;                                    // [deg]
        theta_motor = current_status.theta * 180/PI * (BEAM_LENGTH/LINK_LENGTH) + 90;
float   angle_motor;                                    // angle in [0,256] for Allegro functions
        angle_motor = 64 - theta_motor*(float)128/180; 
float   theta_motor_true;                               // Theta true value (0 -> link is horizontal)
        theta_motor_true = (theta_motor - 90)*PI/180;         

        // Ball position in world coordinates [m]
        ball_x = PIVOT_X -PIVOT_DISTANCE*cos(theta_true) - (BEAM_THICKNESS/2 + BALL_RADIUS)*sin(theta_true) + x*cos(theta_true);
        ball_y = PIVOT_Y -PIVOT_DISTANCE*sin(theta_true) + x*sin(theta_true) + (float)BEAM_THICKNESS/2*cos(theta_true) + BALL_RADIUS*cos(theta_true);
        // Ball position in pixel coordinates:        
        ball_x_pixel = world2pixel_x_coordinate(ball_x);
        ball_y_pixel = world2pixel_y_coordinate(ball_y);

        // Another point rigidly attached to the ball, in order to see the rolling inside the graphics:
float   r_2 = BALL_RADIUS/2;                            // distance from the ball center [m]
float   theta_ball;                                     // angular value of the line from ball center to point center [rad]
float   ball_point_x, ball_point_y;
int     ball_point_x_pixel, ball_point_y_pixel;

        theta_ball = x/BALL_RADIUS;

        ball_point_x = ball_x + r_2*cos(theta_ball)*cos(theta_true) + r_2*sin(theta_ball)*sin(theta_true);
        ball_point_y = ball_y + r_2*cos(theta_ball)*sin(theta_true) - r_2*cos(theta_true)*sin(theta_ball);

        ball_point_x_pixel = world2pixel_x_coordinate(ball_point_x);
        ball_point_y_pixel =  world2pixel_y_coordinate(ball_point_y);

        // Motor Chassis Drawing:
        draw_motor_chassis(system_bitmap);
        // Laser Drawing:
        draw_laser(beam_bitmap, x);
        // Beam Drawing:
        draw_beam_bitmap(beam_bitmap);

        // Relative Beam Pivot position:
        pivot_x_beam = meters2pixels(LASER_L1 + LASER_L2 + BEAM_THICKNESS_2 + PIVOT_DISTANCE);
        pivot_y_beam = meters2pixels(BEAM_HEIGHT) - meters2pixels((float)BEAM_THICKNESS/2); 

        // Drawing the rotated beam based on the current theta value from the system status: 
        pivot_scaled_sprite(system_bitmap, beam_bitmap, world2pixel_x_coordinate(PIVOT_X), world2pixel_y_coordinate(PIVOT_Y),
                                                 pivot_x_beam, pivot_y_beam, ftofix(angle), ftofix(scale)); 

        // Ball Drawing:                                                 
        circlefill(system_bitmap, ball_x_pixel, ball_y_pixel, meters2pixels(BALL_RADIUS), border_color);
        circlefill(system_bitmap, ball_x_pixel, ball_y_pixel, meters2pixels(BALL_RADIUS) - border_size, ball_color);
        // Point on the ball:
        circlefill(system_bitmap, ball_point_x_pixel, ball_point_y_pixel, 2, 16);
        // Beam Support Drawing:
        draw_beam_support(system_bitmap);
        // Pivot position of the link relative to the link bitmap:
        pivot_x_link = meters2pixels(LINK_HEIGHT/2);
        pivot_y_link = meters2pixels(LINK_HEIGHT/2);                     
        // Link 2 Drawing:
        draw_link_2(theta_true, theta_motor_true, system_bitmap);
        // Draw the rotated link:
        pivot_scaled_sprite(system_bitmap, link_bitmap, world2pixel_x_coordinate(MOTOR_CENTER_X), world2pixel_y_coordinate(MOTOR_CENTER_Y),
                                       pivot_x_link, pivot_y_link, ftofix(angle_motor), ftofix(scale));
        // Support 2 Drawing:
        draw_beam_support_2(theta_true, system_bitmap); 
        // Ball status Drawing:
        if (sim_speed == NORMAL) {
        draw_ball_status(system_bitmap, current_status);       
        }
        
        // Blit all the drawn components in the screen in one shot, in order to avoid flickering:
        blit(system_bitmap, screen, 0, 0, UPPER_LEFT_SYS_BOX_X, UPPER_LEFT_SYS_BOX_Y, system_bitmap->w, system_bitmap->h);                               
}
//------------------------------------------------------------------------------------------------------------

void draw_Q_table()
{
int     i, actions;                                                                     // Iterators     
int     num_actions = get_n_actions();                                                  // Number of discretized actions
int     num_states = get_n_states();                                                    // Number of discretized states
int     distance = 30;                                                                  // distance in pixel between the two Q-table parts  
int     Q_table_start_x_2 = Q_TABLE_START_X + CELL_SIZE*num_actions + distance;         // x coordinate (in pixels) of the second Q-table part
char    s_num[4];                                                                       // string for the state number
char    a_num[2];                                                                       // string for the action number
char    s_1[] = "Q-TABLE:";                                                             // String for the title
int     space = 6;                                                                      // space from left Q-table box edge 

        //----------------------------------------------------------------------------------------------------
        // EMPTY Q-TABLE DRAWING
        //----------------------------------------------------------------------------------------------------
        // First Table Half:
        for (i=0; i<num_states/2; i++) {
                for (actions=0; actions < num_actions; actions++) {
                        rectfill(screen, (Q_TABLE_START_X + actions*CELL_SIZE), (Q_TABLE_START_Y + i*CELL_SIZE), 
                                (Q_TABLE_START_X + (actions+1)*CELL_SIZE), (Q_TABLE_START_Y + (i+1)*CELL_SIZE), 0);
                        rectfill(screen, ((Q_TABLE_START_X+1) + actions*CELL_SIZE), ((Q_TABLE_START_Y+1) + i*CELL_SIZE), 
                                ((Q_TABLE_START_X-1) + (actions+1)*CELL_SIZE), ((Q_TABLE_START_Y-1) + (i+1)*CELL_SIZE), 15);
                }
        }
        // Second Table Half:
        for (i=0; i<num_states/2; i++) {
                for (actions=0; actions < num_actions; actions++) {
                        rectfill(screen, (Q_table_start_x_2 + actions*CELL_SIZE), (Q_TABLE_START_Y + i*CELL_SIZE), 
                                (Q_table_start_x_2 + (actions+1)*CELL_SIZE), (Q_TABLE_START_Y + (i+1)*CELL_SIZE), 0);
                        rectfill(screen, ((Q_table_start_x_2+1) + actions*CELL_SIZE), ((Q_TABLE_START_Y+1) + i*CELL_SIZE), 
                                ((Q_table_start_x_2-1) + (actions+1)*CELL_SIZE), ((Q_TABLE_START_Y-1) + (i+1)*CELL_SIZE), 15);
                }
        }
        //----------------------------------------------------------------------------------------------------
        // Q-TABLE TITLE
        //----------------------------------------------------------------------------------------------------     
        rectfill(screen, UPPER_LEFT_Q_BOX_X + space, UPPER_LEFT_Q_BOX_Y + space, UPPER_LEFT_Q_BOX_X + space + 70, UPPER_LEFT_Q_BOX_Y + space + text_height(font) + 4, 16);
        rectfill(screen, UPPER_LEFT_Q_BOX_X + space + 2, UPPER_LEFT_Q_BOX_Y + space + 2, UPPER_LEFT_Q_BOX_X + space + 70 + 2, UPPER_LEFT_Q_BOX_Y + space + text_height(font) + 4 + 2, 16);
        rectfill(screen, UPPER_LEFT_Q_BOX_X + space + 1, UPPER_LEFT_Q_BOX_Y + space + 1, UPPER_LEFT_Q_BOX_X + space + 70 - 1, UPPER_LEFT_Q_BOX_Y + space + text_height(font) + 4 - 1, 15);
        textout_ex(screen, font, s_1, UPPER_LEFT_Q_BOX_X + space + 1 + 2, UPPER_LEFT_Q_BOX_Y + space + 1 + 2, 16, -1);
        //----------------------------------------------------------------------------------------------------
        // LEGEND:
        //---------------------------------------------------------------------------------------------------- 
        draw_Q_table_legend();  
        //----------------------------------------------------------------------------------------------------
        // Q-TABLE STATUS AND ACTION TEXT:
        //---------------------------------------------------------------------------------------------------- 
        for (i=0; i<num_states/2; i++) {
             sprintf(s_num, "%d", i);             
             textout_ex(screen, font, s_num, Q_TABLE_START_X - text_length(font, s_num), Q_TABLE_START_Y + i*CELL_SIZE + 1, 15, -1);                     
        }
        for (i=0; i<num_states/2; i++) {
             sprintf(s_num, "%d", (i + num_states/2));             
             textout_ex(screen, font, s_num, Q_table_start_x_2 - text_length(font, s_num), Q_TABLE_START_Y + i*CELL_SIZE + 1, 15, -1);                     
        }
        for (i=0; i<num_actions; i++) {
                sprintf(a_num, "%d", i);
                textout_ex(screen, font, a_num, Q_TABLE_START_X + i*CELL_SIZE + 1, Q_TABLE_START_Y - text_height(font) - 2, 15, -1);
        }
        for (i=0; i<num_actions; i++) {
                sprintf(a_num, "%d", i);
                textout_ex(screen, font, a_num, Q_table_start_x_2 + i*CELL_SIZE + 1, Q_TABLE_START_Y - text_height(font) - 2, 15, -1);
        }
        //----------------------------------------------------------------------------------------------------
        // Q-TABLE SAVE, LOAD BUTTONS:
        //---------------------------------------------------------------------------------------------------- 
        draw_Q_table_button_instructions();              
}
//------------------------------------------------------------------------------------------------------------

void set_graphics_RL_state(int state, int action, int reward, int step, int episode_counter, float average_TD_error)
{       
        // Update all the RL static buffers:
        reward_state = state;
        reward_action = action;
        current_reward = reward;
        rl_step = step;
        rl_episode_counter = episode_counter;
        rl_step_flag = 1; 
        avg_TD_error = average_TD_error;
}
//------------------------------------------------------------------------------------------------------------

void update_Q_table_graphics()
{

int     n_actions = get_n_actions();                    // Number of discretized actions
int     n_states = get_n_states();                      // Number of discretized states
int     distance = 30;                                  // distance in pixel between the two Q-table parts  
int     Q_table_start_x_2 = Q_TABLE_START_X + CELL_SIZE*n_actions + distance;
// Colors number from 8-bit VGA palette:
int     current_state_color = 32;
int     reward_color = 0;

        //----------------------------------------------------------------------------------------------------
        // UPDATE Q-TABLE DRAWING:
        //----------------------------------------------------------------------------------------------------
        if (rl_step_flag) {

                // Erasing (white coloring) the previous status inside the Q-table:
                if (reward_state_old < (n_states/2)) {
                        rectfill(screen, ((Q_TABLE_START_X+1) + reward_action_old*CELL_SIZE), ((Q_TABLE_START_Y+1) + reward_state_old*CELL_SIZE), 
                        ((Q_TABLE_START_X-1) + (reward_action_old+1)*CELL_SIZE), ((Q_TABLE_START_Y-1) + (reward_state_old+1)*CELL_SIZE), makecol8(255,255,255));
                }
                else {  
                        rectfill(screen, ((Q_table_start_x_2+1) + reward_action_old*CELL_SIZE), ((Q_TABLE_START_Y+1) + (reward_state_old - n_states/2)*CELL_SIZE), 
                        ((Q_table_start_x_2-1) + (reward_action_old+1)*CELL_SIZE), ((Q_TABLE_START_Y-1) + (reward_state_old - n_states/2 + 1)*CELL_SIZE), makecol8(255,255,255));
                } 

                // Getting the Q-value for the previous state:
                reward_old = get_Q_value(reward_state_old, reward_action_old);         

                // Reward color:
                if (reward_old < -50)                            reward_color = makecol8(150,0,0);
                if (reward_old >= -50 && reward_old < -30)       reward_color = makecol8(255,0,0);
                if (reward_old >= -30 && reward_old < -15)       reward_color = makecol8(255,165,0);
                if (reward_old >= -15 && reward_old < 0)         reward_color = makecol8(255,255,0);
                if (reward_old == 0)                             reward_color = makecol8(255,255,255);
                if (reward_old > 0 && reward_old < 15)           reward_color = makecol8(144,238,144);
                if (reward_old >= 15 && reward_old < 30)         reward_color = makecol8(0,255,0);
                if (reward_old >= 30 && reward_old < 50)         reward_color = makecol8(50,205,50);
                if (reward_old > 50)                             reward_color = makecol8(0,100,0);

                // Coloring the old reward inside the old state position in the Q-table:
                if (reward_state_old < (n_states/2)) {
                        rectfill(screen, ((Q_TABLE_START_X+1) + reward_action_old*CELL_SIZE), ((Q_TABLE_START_Y+1) + reward_state_old*CELL_SIZE), 
                        ((Q_TABLE_START_X-1) + (reward_action_old+1)*CELL_SIZE), ((Q_TABLE_START_Y-1) + (reward_state_old+1)*CELL_SIZE), reward_color);
                }
                else {  
                        rectfill(screen, ((Q_table_start_x_2+1) + reward_action_old*CELL_SIZE), ((Q_TABLE_START_Y+1) + (reward_state_old - n_states/2)*CELL_SIZE), 
                        ((Q_table_start_x_2-1) + (reward_action_old+1)*CELL_SIZE), ((Q_TABLE_START_Y-1) + (reward_state_old - n_states/2 + 1)*CELL_SIZE), reward_color);
                }
                // Coloring the current status inside the Q-table:
                if (reward_state < (n_states/2)) {
                        rectfill(screen, ((Q_TABLE_START_X+1) + reward_action*CELL_SIZE), ((Q_TABLE_START_Y+1) + reward_state*CELL_SIZE), 
                        ((Q_TABLE_START_X-1) + (reward_action+1)*CELL_SIZE), ((Q_TABLE_START_Y-1) + (reward_state+1)*CELL_SIZE), current_state_color);
                }
                else { 
                        rectfill(screen, ((Q_table_start_x_2+1) + reward_action*CELL_SIZE), ((Q_TABLE_START_Y+1) + (reward_state - n_states/2)*CELL_SIZE), 
                        ((Q_table_start_x_2-1) + (reward_action+1)*CELL_SIZE), ((Q_TABLE_START_Y-1) + (reward_state - n_states/2 + 1)*CELL_SIZE), current_state_color);
                }        
                // Updating the old values to the new ones:
                reward_state_old = reward_state;
                reward_action_old = reward_action;
        }

}
//------------------------------------------------------------------------------------------------------------

void draw_Q_table_after_loading()
{
int     i, actions;        
int     num_actions = get_n_actions();                  // Number of discretized actions
int     num_states = get_n_states();                    // Number of discretized states
int     distance = 30;                                  // distance in pixel between the two Q-table parts  
int     Q_table_start_x_2 = Q_TABLE_START_X + CELL_SIZE*num_actions + distance;  // x pixel start position of the second half of the Q-table 
int     reward = 0;                                     // reward value buffer
int     reward_color = 0;                               // reward color buffer

        //----------------------------------------------------------------------------------------------------
        // EMPTY Q-TABLE DRAWING
        //----------------------------------------------------------------------------------------------------
        // First Table Half:
        for (i=0; i<num_states/2; i++) {
                for (actions=0; actions < num_actions; actions++) {

                        reward = get_Q_value(i, actions);       // Getting the Q value of the (i, actions) entry of the matrix
                        // Reward color:
                        if (reward < -50)                        reward_color = makecol8(150,0,0);
                        if (reward >= -50 && reward < -30)       reward_color = makecol8(255,0,0);
                        if (reward >= -30 && reward < -15)       reward_color = makecol8(255,165,0);
                        if (reward >= -15 && reward < 0)         reward_color = makecol8(255,255,0);
                        if (reward == 0)                         reward_color = makecol8(255,255,255);
                        if (reward > 0 && reward < 15)           reward_color = makecol8(144,238,144);
                        if (reward >= 15 && reward < 30)         reward_color = makecol8(0,255,0);
                        if (reward >= 30 && reward < 50)         reward_color = makecol8(50,205,50);
                        if (reward > 50)                         reward_color = makecol8(0,100,0);
                        // Coloring the (i, actions) cell:
                        rectfill(screen, ((Q_TABLE_START_X+1) + actions*CELL_SIZE), ((Q_TABLE_START_Y+1) + i*CELL_SIZE), 
                                ((Q_TABLE_START_X-1) + (actions+1)*CELL_SIZE), ((Q_TABLE_START_Y-1) + (i+1)*CELL_SIZE), reward_color);
                }
        }
        // Second Table Half:
        for (i=(num_states/2); i<num_states; i++) {
                for (actions=0; actions < num_actions; actions++) {

                        reward = get_Q_value(i, actions);       // Getting the Q value of the (i, actions) entry of the matrix
                        // Reward color:
                        if (reward < -50)                        reward_color = makecol8(150,0,0);
                        if (reward >= -50 && reward < -30)       reward_color = makecol8(255,0,0);
                        if (reward >= -30 && reward < -15)       reward_color = makecol8(255,165,0);
                        if (reward >= -15 && reward < 0)         reward_color = makecol8(255,255,0);
                        if (reward == 0)                         reward_color = makecol8(255,255,255);
                        if (reward > 0 && reward < 15)           reward_color = makecol8(144,238,144);
                        if (reward >= 15 && reward < 30)         reward_color = makecol8(0,255,0);
                        if (reward >= 30 && reward < 50)         reward_color = makecol8(50,205,50);
                        if (reward > 50)                         reward_color = makecol8(0,100,0);
                        // Coloring the (i, actions) cell:
                        rectfill(screen, ((Q_table_start_x_2+1) + actions*CELL_SIZE), ((Q_TABLE_START_Y+1) + (i + 1 -num_states/2 - 1)*CELL_SIZE), 
                                ((Q_table_start_x_2-1) + (actions+1)*CELL_SIZE), ((Q_TABLE_START_Y-1) + (i+2-num_states/2 - 1)*CELL_SIZE), reward_color);
                }
        }
}
//------------------------------------------------------------------------------------------------------------

void draw_menu_arrows()
{
int     i;      // Iterator
// Keyboard button drawing colors:
int     col_1 = 29;
int     col_2 = 27;
int     col_3 = 20;
int     col_4 = 24;
// Keyboard button points:
int     key_A_x, key_A_y, key_B_x, key_B_y, key_C_x, key_C_y, key_D_x, key_D_y;
int     key_E_x, key_E_y, key_F_x, key_F_y, key_G_x, key_G_y, key_H_x, key_H_y;

        for (i=1; i <= 4; i++) {
            switch (i) {
               case 1:
                  // UP ARROW:
                  key_A_x = UPPER_LEFT_MENU_BOX_X + 135;
                  key_A_y = LOWER_RIGHT_MENU_BOX_Y - 115;
                  break;
               case 2:
                  // LEFT ARROW:
                  key_A_x = UPPER_LEFT_MENU_BOX_X + 135 - KEY_D2;
                  key_A_y = LOWER_RIGHT_MENU_BOX_Y - 115 + KEY_D1;
                  break;
               case 3:
                  // DOWN ARROW:
                  key_A_x = UPPER_LEFT_MENU_BOX_X + 135;
                  key_A_y = LOWER_RIGHT_MENU_BOX_Y - 115 + KEY_D1;
                  break;  
               case 4:
                  // RIGHT ARROW:
                  key_A_x = UPPER_LEFT_MENU_BOX_X + 135 + KEY_D2;
                  key_A_y = LOWER_RIGHT_MENU_BOX_Y - 115 + KEY_D1;
                  break;
               default: break;   
            }            
            // Points definition:
            key_B_x = key_A_x - KEY_L1;
            key_B_y = key_A_y + KEY_L2;
            key_C_x = key_A_x;
            key_C_y = key_A_y + KEY_SIZE;
            key_D_x = key_A_x - KEY_L1;
            key_D_y = key_A_y + KEY_SIZE + KEY_L3;
            key_E_x = key_A_x + KEY_SIZE;
            key_E_y = key_A_y + KEY_SIZE;
            key_F_x = key_A_x + KEY_SIZE + KEY_L1;
            key_F_y = key_A_y + KEY_SIZE + KEY_L3;
            key_G_x = key_A_x + KEY_SIZE + KEY_L3;
            key_G_y = key_A_y + KEY_L2;
            key_H_x = key_A_x + KEY_SIZE;
            key_H_y = key_A_y;            
            // Key buttons drawing:
            int polygon_1[8] = {key_A_x, key_A_y, key_C_x, key_C_y, key_E_x, key_E_y, key_H_x, key_H_y};
            int polygon_2[8] = {key_B_x, key_B_y, key_D_x, key_D_y, key_C_x, key_C_y, key_A_x, key_A_y};
            int polygon_3[8] = {key_H_x, key_H_y, key_E_x, key_E_y, key_F_x, key_F_y, key_G_x, key_G_y};
            int polygon_4[8] = {key_C_x, key_C_y, key_D_x, key_D_y, key_F_x, key_F_y, key_E_x, key_E_y};
            polygon(screen, 4, polygon_1, col_1);
            polygon(screen, 4, polygon_2, col_2);
            polygon(screen, 4, polygon_3, col_3);
            polygon(screen, 4, polygon_4, col_4);
            // Some 3D light
            line(screen, (key_C_x+1), (key_C_y-1), (key_E_x-1), (key_E_y-1), 15);
            line(screen, (key_A_x+1), (key_A_y+1), (key_C_x+1), (key_C_y-1), 15);
            // Arrows and text drawing:
            if (i == 1) {
               // Up Arrow Drawing:
               rectfill(screen, (key_A_x + KEY_SIZE/2 - ARROW_THICKNESS/2), key_A_y + 10, key_A_x + KEY_SIZE/2 + ARROW_THICKNESS/2, key_A_y + 24, 20);
               int   triangle_up[6] = {(key_A_x + KEY_SIZE/2 - TRIANGLE_THICKNESS/2), (key_A_y + 10), (key_A_x + KEY_SIZE/2), (key_A_y + 4), (key_A_x + KEY_SIZE/2 + TRIANGLE_THICKNESS/2), (key_A_y + 10)};
               polygon(screen, 3, triangle_up, 20);
               // Text Contour:
               rectfill(screen, key_A_x  + KEY_SIZE/2 - KEY_TEXT_WIDTH/2, key_A_y - 8 - KEY_TEXT_HEIGHT, key_A_x + KEY_SIZE/2 + KEY_TEXT_WIDTH/2, key_A_y - 8, 16);
               // Text Shadow:
               rectfill(screen, key_A_x  + KEY_SIZE/2 - KEY_TEXT_WIDTH/2 + 2, key_A_y - 8 - KEY_TEXT_HEIGHT + 2, key_A_x + KEY_SIZE/2 + KEY_TEXT_WIDTH/2 + 2, key_A_y - 8 + 2, 16);
               // Text white fill:
               rectfill(screen, (key_A_x + KEY_SIZE/2 - KEY_TEXT_WIDTH/2 + 1), (key_A_y - 8 - KEY_TEXT_HEIGHT + 1), (key_A_x + KEY_SIZE/2 + KEY_TEXT_WIDTH/2 - 1), (key_A_y - 8 - 1), 15);
               textout_centre_ex(screen, font, "ANGLE: +5°", key_A_x + KEY_SIZE/2, key_A_y - 8 - KEY_TEXT_HEIGHT + 3, 16, -1);
               circlefill(screen, key_A_x + KEY_SIZE/2 + 7, key_A_y + 3, 2, 16);
               line(screen, key_A_x + KEY_SIZE/2 + 7, key_A_y + 3, key_A_x + KEY_SIZE/2 + 7, key_A_y - 8, 16);
            }
            if (i == 2) {
               // Left Arrow Drawing:
               rectfill(screen, (key_A_x + 10), (key_A_y + KEY_SIZE/2 - ARROW_THICKNESS/2), (key_A_x + 24), (key_A_y + KEY_SIZE/2 + ARROW_THICKNESS/2), 20);
               int   triangle_left[6] = {(key_A_x + 10), (key_A_y + KEY_SIZE/2 + TRIANGLE_THICKNESS/2), (key_A_x + 4), (key_A_y + KEY_SIZE/2), (key_A_x + 10), (key_A_y + KEY_SIZE/2 - TRIANGLE_THICKNESS/2)};
               polygon(screen, 3, triangle_left, 20);
               // Text Contour:
               rectfill(screen, key_A_x  + KEY_SIZE/2 + 28 - KEY_TEXT_WIDTH, key_A_y - 8 - KEY_TEXT_HEIGHT, key_A_x + KEY_SIZE/2 + 28, key_A_y - 8, 16);
               // Text Shadow:
               rectfill(screen, key_A_x  + KEY_SIZE/2 + 28 - KEY_TEXT_WIDTH + 2, key_A_y - 8 - KEY_TEXT_HEIGHT + 2, key_A_x + KEY_SIZE/2 + 28 + 2, key_A_y - 8 + 2, 16);
               // Text white fill:
               rectfill(screen, key_A_x  + KEY_SIZE/2 + 28 - KEY_TEXT_WIDTH + 1, key_A_y - 8 - KEY_TEXT_HEIGHT + 1, key_A_x + KEY_SIZE/2 + 28 - 1, key_A_y - 8 - 1, 15);
               textout_centre_ex(screen, font, "LEFT PUSH", key_A_x + KEY_SIZE/2 - KEY_TEXT_WIDTH/2 + 28, key_A_y - 8 - KEY_TEXT_HEIGHT + 3, 16, -1);
               circlefill(screen, key_A_x + KEY_SIZE/2, key_A_y + 3, 2, 16);
               line(screen, key_A_x + KEY_SIZE/2, key_A_y + 3, key_A_x + KEY_SIZE/2 - KEY_TEXT_WIDTH/2 + 28, key_A_y - 8, 16);
            }
            if (i == 3) {
               // Down Arrow Drawing:
               rectfill(screen, (key_A_x + KEY_SIZE/2 - ARROW_THICKNESS/2), (key_A_y + 6), (key_A_x + KEY_SIZE/2 + ARROW_THICKNESS/2), (key_A_y + 20), 20);
               int   triangle_down[6] = {(key_A_x + KEY_SIZE/2 - TRIANGLE_THICKNESS/2), (key_A_y + 20), (key_A_x + KEY_SIZE/2), (key_A_y + 26), (key_A_x + KEY_SIZE/2 + TRIANGLE_THICKNESS/2), (key_A_y + 20)};
               polygon(screen, 3, triangle_down, 20);
               // Text Contour:
               rectfill(screen, key_A_x  + KEY_SIZE/2 - KEY_TEXT_WIDTH/2, key_A_y + 8 + KEY_SIZE + KEY_L3, key_A_x + KEY_SIZE/2 + KEY_TEXT_WIDTH/2, key_A_y + KEY_TEXT_HEIGHT + KEY_SIZE + KEY_L3 + 8, 16);
               // Text Shadow:
               rectfill(screen, key_A_x  + KEY_SIZE/2 - KEY_TEXT_WIDTH/2 + 2, key_A_y + 8 + KEY_SIZE + KEY_L3 + 2, key_A_x + KEY_SIZE/2 + KEY_TEXT_WIDTH/2 + 2, key_A_y + KEY_TEXT_HEIGHT + KEY_SIZE + KEY_L3 + 8 + 2, 16);
               // Text white fill:
               rectfill(screen, (key_A_x + KEY_SIZE/2 - KEY_TEXT_WIDTH/2 + 1), (key_A_y + 8 + KEY_SIZE + KEY_L3 + 1), (key_A_x + KEY_SIZE/2 + KEY_TEXT_WIDTH/2 - 1), (key_A_y + 8 + KEY_TEXT_HEIGHT + KEY_SIZE + KEY_L3 - 1), 15);
               textout_centre_ex(screen, font, "ANGLE: -5°", key_A_x + KEY_SIZE/2, key_A_y + KEY_SIZE + KEY_L3 + 8 + 3, 16, -1);
               circlefill(screen, key_A_x + KEY_SIZE/2 - 7, key_A_y + KEY_SIZE - 4, 2, 16);
               line(screen, key_A_x + KEY_SIZE/2 - 7, key_A_y + KEY_SIZE - 4, key_A_x + KEY_SIZE/2 - 7, key_A_y + KEY_SIZE + KEY_L3 + 8, 16);
            }
            if (i == 4) {
               // Right Arrow Drawing:
               rectfill(screen, (key_A_x + 6), (key_A_y + KEY_SIZE/2 - ARROW_THICKNESS/2), (key_A_x + 20), (key_A_y + KEY_SIZE/2 + ARROW_THICKNESS/2), 20);
               int   triangle_right[6] = {(key_A_x + 20), (key_A_y + KEY_SIZE/2 - TRIANGLE_THICKNESS/2), (key_A_x + 26), (key_A_y + KEY_SIZE/2), (key_A_x + 20), (key_A_y + KEY_SIZE/2 + TRIANGLE_THICKNESS/2)};
               polygon(screen, 3, triangle_right, 20);
               // Text Contour:
               rectfill(screen, key_A_x  + KEY_SIZE/2 + 58 - KEY_TEXT_WIDTH, key_A_y - 8 - KEY_TEXT_HEIGHT, key_A_x + KEY_SIZE/2 + 58, key_A_y - 8, 16);
               // Text Shadow:
               rectfill(screen, key_A_x  + KEY_SIZE/2 + 58 - KEY_TEXT_WIDTH + 2, key_A_y - 8 - KEY_TEXT_HEIGHT + 2, key_A_x + KEY_SIZE/2 + 58 + 2, key_A_y - 8 + 2, 16);
               // Text white fill:
               rectfill(screen, key_A_x  + KEY_SIZE/2 + 58 - KEY_TEXT_WIDTH + 1, key_A_y - 8 - KEY_TEXT_HEIGHT + 1, key_A_x + KEY_SIZE/2 + 58 - 1, key_A_y - 8 - 1, 15);
               textout_centre_ex(screen, font, "RIGHT PUSH", key_A_x + KEY_SIZE/2 - KEY_TEXT_WIDTH/2 + 58, key_A_y - 8 - KEY_TEXT_HEIGHT + 3, 16, -1);
               circlefill(screen, key_A_x + KEY_SIZE/2, key_A_y + 3, 2, 16);
               line(screen, key_A_x + KEY_SIZE/2, key_A_y + 3, key_A_x + KEY_SIZE/2 - KEY_TEXT_WIDTH/2 + 58, key_A_y - 8, 16);
            }
        }       
}
//------------------------------------------------------------------------------------------------------------

void draw_RL_modes_text()
{
// Distance from menu border:
int     dist_1 = 28;
int     dist_2 = 10;
int     dist_3 = 12;
// Triangle dimensions:
int     triangle_height = 8;
int     triangle_width = 10;
// Keyboard button points:
int     key_A_x, key_A_y;
// Triangle vertix positions:
int     P1_x, P1_y, P2_x, P2_y, P3_x, P3_y;
// Text strings:
char    s1_1[] = "PRESS:";
char    s1_2[] = "TO START RL ";
char    s2_2[] = "TO PLAY YOURSELF ";
char    s3_2[] = "TO EXPLOIT Q-MATRIX ";
char    s4_2[] = "TO QUIT";

        // Mode Text Contour:
        rectfill(screen, UPPER_LEFT_MENU_BOX_X + 1, UPPER_LEFT_MENU_BOX_Y + 1, 
                UPPER_LEFT_MENU_BOX_X + MENU_BOX_SEPARATOR - 1, UPPER_LEFT_MENU_BOX_Y + 110, 0);
        rectfill(screen, UPPER_LEFT_MENU_BOX_X + 1 + 2, UPPER_LEFT_MENU_BOX_Y + 1 + 2, 
                UPPER_LEFT_MENU_BOX_X + MENU_BOX_SEPARATOR - 1 - 2, UPPER_LEFT_MENU_BOX_Y + 110 - 2, 15);
        //----------------------------------------------------------------------------------------------------
        // First Mode:
        // Triangle:
        P1_x = UPPER_LEFT_MENU_BOX_X + dist_3;
        P1_y = UPPER_LEFT_MENU_BOX_Y + dist_3;
        P2_x = P1_x;
        P2_y = P1_y + triangle_height;
        P3_x = P1_x + triangle_width;
        P3_y = P1_y + triangle_height/2;
        triangle(screen, P1_x, P1_y, P2_x, P2_y, P3_x, P3_y, 16);
        // Text:
        textout_ex(screen, font, s1_1, UPPER_LEFT_MENU_BOX_X + dist_1 + 2, UPPER_LEFT_MENU_BOX_Y + dist_2 + 3, 16, -1);        
        // Button position:
        key_A_x = UPPER_LEFT_MENU_BOX_X + dist_1 + text_length(font, s1_1) + 15;
        key_A_y = UPPER_LEFT_MENU_BOX_Y + dist_2;

        draw_button_key(key_A_x, key_A_y, "R");
        // Text after button:
        textout_ex(screen, font, s1_2, key_A_x + KEY_SIZE_2 + 12 + 2, UPPER_LEFT_MENU_BOX_Y + dist_2 + 3, 16, -1);
        //----------------------------------------------------------------------------------------------------
        // Second Mode:
        // Triangle:
        P1_x = UPPER_LEFT_MENU_BOX_X + dist_3;
        P1_y = UPPER_LEFT_MENU_BOX_Y + dist_3 + 25;
        P2_x = P1_x;
        P2_y = P1_y + triangle_height;
        P3_x = P1_x + triangle_width;
        P3_y = P1_y + triangle_height/2;
        triangle(screen, P1_x, P1_y, P2_x, P2_y, P3_x, P3_y, 16);
        // Text:
        textout_ex(screen, font, s1_1, UPPER_LEFT_MENU_BOX_X + dist_1 + 2, UPPER_LEFT_MENU_BOX_Y + dist_2 + 3 + 25, 16, -1);        
        // Button position:
        key_A_x = UPPER_LEFT_MENU_BOX_X + dist_1 + text_length(font, s1_1) + 15;
        key_A_y = UPPER_LEFT_MENU_BOX_Y + dist_2 + 25;

        draw_button_key(key_A_x, key_A_y, "P");
        // Text after button:
        textout_ex(screen, font, s2_2, key_A_x + KEY_SIZE_2 + 12 + 2, UPPER_LEFT_MENU_BOX_Y + dist_2 + 3 + 25, 16, -1);
        //----------------------------------------------------------------------------------------------------
        // Third Mode:
        // Triangle:
        P1_x = UPPER_LEFT_MENU_BOX_X + dist_3;
        P1_y = UPPER_LEFT_MENU_BOX_Y + dist_3 + 50;
        P2_x = P1_x;
        P2_y = P1_y + triangle_height;
        P3_x = P1_x + triangle_width;
        P3_y = P1_y + triangle_height/2;
        triangle(screen, P1_x, P1_y, P2_x, P2_y, P3_x, P3_y, 16);
        // Text:
        textout_ex(screen, font, s1_1, UPPER_LEFT_MENU_BOX_X + dist_1 + 2, UPPER_LEFT_MENU_BOX_Y + dist_2 + 3 + 50, 16, -1);        
        // Button position:
        key_A_x = UPPER_LEFT_MENU_BOX_X + dist_1 + text_length(font, s1_1) + 15;
        key_A_y = UPPER_LEFT_MENU_BOX_Y + dist_2 + 50;

        draw_button_key(key_A_x, key_A_y, "E");
        // Text after button:
        textout_ex(screen, font, s3_2, key_A_x + KEY_SIZE_2 + 12 + 2, UPPER_LEFT_MENU_BOX_Y + dist_2 + 3 + 50, 16, -1);
        //----------------------------------------------------------------------------------------------------
        // Exit:
        // Triangle:
        P1_x = UPPER_LEFT_MENU_BOX_X + dist_3;
        P1_y = UPPER_LEFT_MENU_BOX_Y + dist_3 + 75;
        P2_x = P1_x;
        P2_y = P1_y + triangle_height;
        P3_x = P1_x + triangle_width;
        P3_y = P1_y + triangle_height/2;
        triangle(screen, P1_x, P1_y, P2_x, P2_y, P3_x, P3_y, 16);
        // Text:
        textout_ex(screen, font, s1_1, UPPER_LEFT_MENU_BOX_X + dist_1 + 2, UPPER_LEFT_MENU_BOX_Y + dist_2 + 3 + 75, 16, -1);        
        // Button position:
        key_A_x = UPPER_LEFT_MENU_BOX_X + dist_1 + text_length(font, s1_1) + 15;
        key_A_y = UPPER_LEFT_MENU_BOX_Y + dist_2 + 75;

        draw_button_key(key_A_x, key_A_y, "Q");
        // Text after button:
        textout_ex(screen, font, s4_2, key_A_x + KEY_SIZE_2 + 12 + 2, UPPER_LEFT_MENU_BOX_Y + dist_2 + 3 + 75, 16, -1);
}
//------------------------------------------------------------------------------------------------------------

int theta_2_plot_pixels(float theta)
{

int     plot_y_pixel;                                           // y position in the screen of the current theta value
float   theta_step_rad, theta_max;

        theta_step_rad = THETA_STEP * PI / 180;                 // theta step in [rad]
        theta_max = ((NUM_ACTIONS - 1) / 2) * theta_step_rad;   // theta max in [rad]
        

        plot_y_pixel = -theta*(PLOT_HEIGHT/2 - 5)/theta_max;    // The y pixel grows to lower window parts! Hence the minus sign
        plot_y_pixel = plot_y_pixel + UPPER_LEFT_PLOT_Y + PLOT_HEIGHT/2;

        return plot_y_pixel;
}
//------------------------------------------------------------------------------------------------------------

void update_theta_plot(struct status current_status)
{

int             plot_y_pixel, plot_x_pixel, theta_ref_y_pixel;          // Plot y coordinates in pixels
float           theta, theta_deg, theta_ref;                            // Current system theta value buffers
char            s[20];                                                  // string buffer
                
                // If the theta plot reached the window limit, then clear the plots and start from the left edge again:
                if (plot_step == PLOT_WIDTH/2) {
                        // Clear the plot:
                        rectfill(screen, UPPER_LEFT_PLOT_X, UPPER_LEFT_PLOT_Y,
                                        LOWER_RIGHT_PLOT_X, LOWER_RIGHT_PLOT_Y,
                                        PLOT_BKG);
                        // Zero green line drawing:
                        line(screen, UPPER_LEFT_PLOT_X, UPPER_LEFT_PLOT_Y + PLOT_HEIGHT/2, UPPER_LEFT_PLOT_X + PLOT_WIDTH, UPPER_LEFT_PLOT_Y + PLOT_HEIGHT/2, 46);                
                        plot_x_pixel_old = UPPER_LEFT_PLOT_X;
                        plot_step = 0;                 
                }
                // Getting the current theta value:
                theta = current_status.theta;
                plot_y_pixel = theta_2_plot_pixels(theta);              // converting theta value to y pixel inside theta plot
                // Draw every new theta value two pixels to the right:
                plot_x_pixel = (plot_step*2);  
                plot_x_pixel = plot_x_pixel + UPPER_LEFT_PLOT_X; 
                // Connect the last two theta pixel representation with a line:
                line(screen, plot_x_pixel_old, plot_y_pixel_old, plot_x_pixel, plot_y_pixel, makecol8(255,255,0));
                // Theta value text:                
                theta_deg = theta * 180/PI;
                sprintf(s, "Theta: %.1f [deg]", theta_deg);
                textout_ex(screen, font, s, UPPER_LEFT_PLOT_X + 180, UPPER_LEFT_PLOT_Y - text_height(font) - 4, 16, -1);
                // Theta Reference to the motor drawing:
                theta_ref = current_status.theta_ref; 
                theta_ref_y_pixel = theta_2_plot_pixels(theta_ref);
                // Draw the theta ref once every RL step:
                if (rl_step_flag) {
                        if (plot_x_pixel <= LOWER_RIGHT_PLOT_X - 30) {                        
                                line(screen, plot_x_pixel, theta_ref_y_pixel, plot_x_pixel + 30, theta_ref_y_pixel, makecol8(255,0,0));
                                // why + 30? the RL task in NORMAL mode makes a step every 450 ms, the graphics task instead makes a step
                                // every 30 ms. Since we are drawing 2 pixels to the right every graphic task step, the next RL step
                                // will be 450/30*2 (=30) pixels to the right
                        }
                        else {
                                line(screen, plot_x_pixel, theta_ref_y_pixel, LOWER_RIGHT_PLOT_X, theta_ref_y_pixel, makecol8(255,0,0));  
                        }
                }
                // Updating the old pixel values:
                plot_x_pixel_old = plot_x_pixel;
                plot_y_pixel_old = plot_y_pixel;
                plot_step++;
}
//------------------------------------------------------------------------------------------------------------

void draw_motor_chassis(BITMAP* system_bitmap)
{
// Motor chassis external rectangle positions in pixels:
int     motor_upper_left_x, motor_upper_left_y;
int     motor_lower_rigth_x, motor_lower_right_y;
int     border_size = 2;
// Some distances for the drawing:
int     d1, r1, r2;
        d1 = 10;
        r1 = 22;
        r2 = 6;

        motor_upper_left_x = world2pixel_x_coordinate(MOTOR_CENTER_X - MOTOR_SIZE/2);
        motor_upper_left_y = world2pixel_y_coordinate(MOTOR_CENTER_Y + MOTOR_SIZE/2);
        motor_lower_rigth_x = world2pixel_x_coordinate(MOTOR_CENTER_X + MOTOR_SIZE/2);
        motor_lower_right_y = world2pixel_y_coordinate(MOTOR_CENTER_Y - MOTOR_SIZE/2);

        // Contour:
        rectfill(system_bitmap, motor_upper_left_x, motor_upper_left_y, motor_lower_rigth_x, motor_lower_right_y, 16);
        // Internal part:
        rectfill(system_bitmap, motor_upper_left_x + border_size, motor_upper_left_y + border_size, motor_lower_rigth_x - border_size, motor_lower_right_y - border_size, MOTOR_COL);

        // Little chamfers at the external vertices:
        triangle(system_bitmap, motor_upper_left_x, motor_upper_left_y + 4, motor_upper_left_x, motor_upper_left_y, motor_upper_left_x + 4, motor_upper_left_y, 15);
        line(system_bitmap, motor_upper_left_x, motor_upper_left_y + 4, motor_upper_left_x + 4, motor_upper_left_y, 16);
        line(system_bitmap, motor_upper_left_x, motor_upper_left_y + 4 + 1, motor_upper_left_x + 4 + 1, motor_upper_left_y, 16);

        triangle(system_bitmap, motor_upper_left_x, motor_lower_right_y -4, motor_upper_left_x, motor_lower_right_y, motor_upper_left_x + 4, motor_lower_right_y, 15);
        line(system_bitmap, motor_upper_left_x, motor_lower_right_y -4, motor_upper_left_x + 4, motor_lower_right_y, 16);
        line(screen, motor_upper_left_x, motor_lower_right_y -4 - 1, motor_upper_left_x + 4 + 1, motor_lower_right_y, 16);

        
        triangle(system_bitmap, motor_lower_rigth_x - 4, motor_upper_left_y, motor_lower_rigth_x, motor_upper_left_y, motor_lower_rigth_x, motor_upper_left_y + 4, 15);
        line(system_bitmap, motor_lower_rigth_x - 4, motor_upper_left_y , motor_lower_rigth_x, motor_upper_left_y + 4, 16);
        line(system_bitmap, motor_lower_rigth_x - 4 - 1, motor_upper_left_y , motor_lower_rigth_x, motor_upper_left_y + 4 + 1, 16);

        triangle(system_bitmap, motor_lower_rigth_x - 4, motor_lower_right_y, motor_lower_rigth_x, motor_lower_right_y, motor_lower_rigth_x, motor_lower_right_y - 4, 15);
        line(system_bitmap, motor_lower_rigth_x - 4, motor_lower_right_y, motor_lower_rigth_x,  motor_lower_right_y - 4, 16);
        line(system_bitmap, motor_lower_rigth_x - 4 - 1, motor_lower_right_y, motor_lower_rigth_x,  motor_lower_right_y - 4 - 1, 16);        

        // Middle circle:
        circlefill(system_bitmap, world2pixel_x_coordinate(MOTOR_CENTER_X), world2pixel_y_coordinate(MOTOR_CENTER_Y), r1, 16);
        circlefill(system_bitmap, world2pixel_x_coordinate(MOTOR_CENTER_X), world2pixel_y_coordinate(MOTOR_CENTER_Y), r1 - border_size, 27);

        // Other 4 circles:
        circlefill(system_bitmap, motor_upper_left_x + d1, motor_upper_left_y + d1, r2, 16);
        circlefill(system_bitmap, motor_upper_left_x + d1, motor_upper_left_y + d1, r2 - border_size, 27);

        circlefill(system_bitmap, motor_upper_left_x + d1, motor_lower_right_y - d1, r2, 16);
        circlefill(system_bitmap, motor_upper_left_x + d1, motor_lower_right_y - d1, r2 - border_size, 27);

        circlefill(system_bitmap, motor_lower_rigth_x - d1, motor_lower_right_y - d1, r2, 16);
        circlefill(system_bitmap, motor_lower_rigth_x - d1, motor_lower_right_y - d1, r2 - border_size, 27);

        circlefill(system_bitmap, motor_lower_rigth_x - d1, motor_upper_left_y + d1, r2, 16);
        circlefill(system_bitmap, motor_lower_rigth_x - d1, motor_upper_left_y + d1, r2 - border_size, 27);
}
//------------------------------------------------------------------------------------------------------------

void draw_laser(BITMAP* beam_bitmap, float x)
{

// Some Pixel lengths for the drawing:
int     h1 = 3; 
int     h2 = 4; 
int     h3 = 3; 
int     h4 = 6; 
int     l3 = meters2pixels(LASER_L1) - 4;
int     border_size = 2;
int     laser_support_col = 6; 
int     border_color = 16;
int     laser_y_center, laser_x_center;

        // Laser Support:
        rectfill(beam_bitmap, 0, meters2pixels(BEAM_HEIGHT/2) + h1, meters2pixels(LASER_L1), meters2pixels(BEAM_HEIGHT) - h1, border_color);
        rectfill(beam_bitmap, 0 + border_size, meters2pixels(BEAM_HEIGHT/2) + h1 + border_size, meters2pixels(LASER_L1) - border_size, meters2pixels(BEAM_HEIGHT) - h1 - border_size, laser_support_col);

        // Laser support rigid link:
        rectfill(beam_bitmap, meters2pixels(LASER_L1), meters2pixels(BEAM_HEIGHT/2) + h1 + h2, meters2pixels(LASER_L1 + LASER_L2), meters2pixels(BEAM_HEIGHT/2) + h1 + h2 + h3, 16);

        // Laser box:
        rectfill(beam_bitmap, 2, meters2pixels(BEAM_HEIGHT/2) - h3/2 - h2 - h4, 2 + l3, meters2pixels(BEAM_HEIGHT/2) + h1, 16);
        rectfill(beam_bitmap, 2 + l3, meters2pixels(BEAM_HEIGHT/2) - h3/2 - h2 - h4 + 2, 2 + l3 + 3,  meters2pixels(BEAM_HEIGHT/2) - h3/2 - h2 - h4 + 2 + 7, 16);

        laser_y_center =  meters2pixels(BEAM_HEIGHT/2) - h3/2 - h2 - h4 + 5;
        laser_x_center = 2 + l3 + 4;

        // Laser line:
        line(beam_bitmap, laser_x_center, laser_y_center, laser_x_center + meters2pixels(BEAM_THICKNESS_2) + meters2pixels(x), laser_y_center, makecol8(255, 0, 0));      

}
//------------------------------------------------------------------------------------------------------------

void draw_RL_status(BITMAP* rl_status_bitmap)
{

if (rl_step_flag) {

        int     v_space = 9;                            // vertical space between texts (in pixels)
        int     text_2_x;                               // position of 2nd column of text in the screen
                text_2_x = 155;
        // Getting RL status:        
        int     reward = current_reward;
        int     episode = rl_episode_counter;
        int     step = rl_step;
        float   explor_prob = ql_get_epsilon();
        float   disc_factor = ql_get_discount_factor();
        float   learn_rate = ql_get_learning_rate();
        float   expl_decay = ql_get_expl_decay(); 
        float   average_TD_error = avg_TD_error; 
        int     rl_current_mode = get_rl_mode();  
        int     ql_policy = get_ql_policy();
        float   beta_boltz = get_beta_boltz();    

        char    s_1[25], s_2[25], s_3[25], s_4[25], s_5[25], s_6[25], s_7[25], s_8[25], s_12[30], s_15[25];
        char    s_9[] = "CURRENT RL MODE: PLAY";
        char    s_10[] = "CURRENT RL MODE: EXPLOIT";
        char    s_11[] = "CURRENT RL MODE: LEARNING"; 
        char    s_b[] = "BOLTZMANN";
        char    s_g[] = "EPS-GREEDY";

                sprintf(s_1, "RL EPISODE: %d", episode);
                sprintf(s_2, "RL STEP: %d", step);
                sprintf(s_3, "RL REWARD: %d", reward);
                sprintf(s_4, "EXPLOR. PROB.: %.2f", explor_prob);
                sprintf(s_5, "DISCOUNT FACT.: %.2f", disc_factor);
                sprintf(s_6, "LEARNING RATE: %.2f", learn_rate);
                sprintf(s_7, "EXPLOR. DECAY: %.2f", expl_decay);
                sprintf(s_8, "AVG TD ERROR: %.1f", average_TD_error);

                sprintf(s_15, "BETA BOLTZ.: %.3f", beta_boltz);

                if (ql_policy == EPS_GREEDY) {
                        sprintf(s_12, "Q-LEARNING POLICY: %s", s_g);
                }
                if (ql_policy == BOLTZMANN) {
                        sprintf(s_12, "Q-LEARNING POLICY: %s", s_b);
                }

                textout_ex(rl_status_bitmap, font, s_1, 4, 5, 16, -1);
                textout_ex(rl_status_bitmap, font, s_2, 4, 5 + text_height(font) + v_space, 16, -1);
                textout_ex(rl_status_bitmap, font, s_3, 4, 5 + 2*text_height(font) + 2*v_space, 16, -1);
                textout_ex(rl_status_bitmap, font, s_8, 4, 5 + 3*text_height(font) + 3*v_space, 16, -1);

                if (rl_current_mode == LEARNING_MODE) {
                        textout_ex(rl_status_bitmap, font, s_11, 4, 5 + 4*text_height(font) + 4*v_space, 16, -1);
                }

                if (rl_current_mode == EXPLOIT_MODE) {
                        textout_ex(rl_status_bitmap, font, s_10, 4, 5 + 4*text_height(font) + 4*v_space, 16, -1);
                }

                if (rl_current_mode == PLAY_MODE) {
                        textout_ex(rl_status_bitmap, font, s_9, 4, 5 + 4*text_height(font) + 4*v_space, 16, -1);
                }

                if (ql_policy == EPS_GREEDY) {
                        textout_ex(rl_status_bitmap, font, s_4, text_2_x, 5, 16, -1);
                }
                if (ql_policy == BOLTZMANN) {
                        textout_ex(rl_status_bitmap, font, s_15, text_2_x, 5, 16, -1);
                }
                textout_ex(rl_status_bitmap, font, s_5, text_2_x, 5 + text_height(font) + v_space, 16, -1);
                textout_ex(rl_status_bitmap, font, s_6, text_2_x, 5 + 2*text_height(font) + 2*v_space, 16, -1);
                textout_ex(rl_status_bitmap, font, s_7, text_2_x, 5 + 3*text_height(font) + 3*v_space, 16, -1);
                // Q-Exploration Policy:
                textout_ex(rl_status_bitmap, font, s_12, 4, 5 + 5*text_height(font) + 5*v_space, 16, -1);

                // Blit to the screen in one shot in order to avoid graphics flickering:
                blit(rl_status_bitmap, screen, 0, 0, UPPER_LEFT_STATUS_X + 2, UPPER_LEFT_STATUS_Y + 2, rl_status_bitmap->w, rl_status_bitmap->h);
}
}
//------------------------------------------------------------------------------------------------------------

void draw_button_key(int A_x, int A_y, char* text)
{
// Keyboard button drawing colors:
int     col_1 = 29;
int     col_2 = 27;
int     col_3 = 20;
int     col_4 = 24;
// Keyboard button points:
int     key_A_x, key_A_y, key_B_x, key_B_y, key_C_x, key_C_y, key_D_x, key_D_y;
int     key_E_x, key_E_y, key_F_x, key_F_y, key_G_x, key_G_y, key_H_x, key_H_y;
        
        // Button position:
        key_A_x = A_x;
        key_A_y = A_y;
        // Points definition:
        key_B_x = key_A_x - KEY_L1_2;
        key_B_y = key_A_y + KEY_L2_2;
        key_C_x = key_A_x;
        key_C_y = key_A_y + KEY_SIZE_2;
        key_D_x = key_A_x - KEY_L1_2;
        key_D_y = key_A_y + KEY_SIZE_2 + KEY_L3_2;
        key_E_x = key_A_x + KEY_SIZE_2;
        key_E_y = key_A_y + KEY_SIZE_2;
        key_F_x = key_A_x + KEY_SIZE_2 + KEY_L1_2;
        key_F_y = key_A_y + KEY_SIZE_2 + KEY_L3_2;
        key_G_x = key_A_x + KEY_SIZE_2 + KEY_L3_2;
        key_G_y = key_A_y + KEY_L2_2;
        key_H_x = key_A_x + KEY_SIZE_2;
        key_H_y = key_A_y;

        int polygon_1[8] = {key_A_x, key_A_y, key_C_x, key_C_y, key_E_x, key_E_y, key_H_x, key_H_y};
        int polygon_2[8] = {key_B_x, key_B_y, key_D_x, key_D_y, key_C_x, key_C_y, key_A_x, key_A_y};
        int polygon_3[8] = {key_H_x, key_H_y, key_E_x, key_E_y, key_F_x, key_F_y, key_G_x, key_G_y};
        int polygon_4[8] = {key_C_x, key_C_y, key_D_x, key_D_y, key_F_x, key_F_y, key_E_x, key_E_y};

        polygon(screen, 4, polygon_1, col_1);
        polygon(screen, 4, polygon_2, col_2);
        polygon(screen, 4, polygon_3, col_3);
        polygon(screen, 4, polygon_4, col_4);

        // Some 3D light
        line(screen, (key_C_x+1), (key_C_y-1), (key_E_x-1), (key_E_y-1), 15);
        line(screen, (key_A_x+1), (key_A_y+1), (key_C_x+1), (key_C_y-1), 15);

        // Letter on the button:
        textout_ex(screen, font, text, key_A_x + 5, key_A_y + 4, col_3, -1);

}
//------------------------------------------------------------------------------------------------------------

void draw_ball_status(BITMAP* system_bitmap, struct status system_state)
{

char            s_x[30];                // x position text buffer
char            s_v[30];                // v velocity text buffer                                                   
char            s_f[30];                // friction coeff. text buffer
int             v_space = 6;            // vertical space between text in pixels
float           friction_coeff = get_friction_coeff();

                sprintf(s_x, "BALL POSITION: %.2f m", system_state.x);
                sprintf(s_v, "BALL VELOCITY: %.2f m/s", system_state.v);
                sprintf(s_f, "FRICTION COEFF.: %.2f", friction_coeff);

                textout_ex(system_bitmap, font, s_x, 10, 30, 16, -1);
                textout_ex(system_bitmap, font, s_v, 10, 30 + text_height(font) + v_space, 16, -1);
                textout_ex(system_bitmap, font, s_f, 10, 30 + 2*text_height(font) + 2*v_space, 16, -1);
}
//------------------------------------------------------------------------------------------------------------

void set_q_table_graphics_flag(int i)
{
        q_table_load_graphics_flag = i;
}
//------------------------------------------------------------------------------------------------------------

void draw_link_bitmap(BITMAP* link_bitmap)
{

int     border_size = 2;                // Size in pixels of black contour of drawing components
int     border_color = 16;             
int     link_color = 28;
// Polygon between semi-circles coordinates:
int     polygon_1[8] = {meters2pixels(LINK_HEIGHT/2), 0, meters2pixels(LINK_LENGTH-LINK_D1), meters2pixels(LINK_HEIGHT/2 - LINK_D1), 
                        meters2pixels(LINK_LENGTH- LINK_D1), meters2pixels(LINK_HEIGHT - LINK_D1), meters2pixels(LINK_HEIGHT/2), meters2pixels(LINK_HEIGHT)};
int     polygon_1_2[8] = {meters2pixels(LINK_HEIGHT/2), 0 + border_size, meters2pixels(LINK_LENGTH-LINK_D1), meters2pixels(LINK_HEIGHT/2 - LINK_D1) + border_size, 
                          meters2pixels(LINK_LENGTH- LINK_D1), meters2pixels(LINK_HEIGHT - LINK_D1) - border_size, meters2pixels(LINK_HEIGHT/2), meters2pixels(LINK_HEIGHT) - border_size};

        // Left Semi-Circle:
        circlefill(link_bitmap, meters2pixels(LINK_HEIGHT/2), meters2pixels(LINK_HEIGHT/2), meters2pixels(LINK_HEIGHT/2), border_color);
        circlefill(link_bitmap, meters2pixels(LINK_HEIGHT/2), meters2pixels(LINK_HEIGHT/2), meters2pixels(LINK_HEIGHT/2) - border_size, link_color);
        // Right Semi-Circle:
        circlefill(link_bitmap, meters2pixels(LINK_LENGTH-LINK_D1), meters2pixels(LINK_HEIGHT/2), meters2pixels(LINK_D1), border_color);
        circlefill(link_bitmap, meters2pixels(LINK_LENGTH-LINK_D1), meters2pixels(LINK_HEIGHT/2), meters2pixels(LINK_D1) - border_size, link_color);
        // Polygon Between Semi-Circle:
        polygon(link_bitmap, 4, polygon_1, border_color);
        polygon(link_bitmap, 4, polygon_1_2, link_color);
        // Central Motor Hinge:
        circlefill(link_bitmap, meters2pixels(LINK_HEIGHT/2), meters2pixels(LINK_HEIGHT/2), meters2pixels(LINK_HEIGHT/4), border_color);
        circlefill(link_bitmap, meters2pixels(LINK_HEIGHT/2), meters2pixels(LINK_HEIGHT/2), meters2pixels(LINK_HEIGHT/4) - border_size, 0);
        // External Hinge:
        circlefill(link_bitmap, meters2pixels(LINK_LENGTH - LINK_D1), meters2pixels(LINK_HEIGHT/2), meters2pixels(LINK_D1/(2)), border_color);
        circlefill(link_bitmap, meters2pixels(LINK_LENGTH - LINK_D1), meters2pixels(LINK_HEIGHT/2), meters2pixels(LINK_D1/(2)) - border_size, 0);
        // Other Hinge:
        circlefill(link_bitmap, meters2pixels(LINK_HEIGHT/2 + LINK_LENGTH/(3.5)), meters2pixels(LINK_HEIGHT/2), meters2pixels(LINK_HEIGHT/5), border_color);
        circlefill(link_bitmap, meters2pixels(LINK_HEIGHT/2 + LINK_LENGTH/(3.5)), meters2pixels(LINK_HEIGHT/2), meters2pixels(LINK_HEIGHT/5) - border_size, 0);
        // Other Hinge:
        circlefill(link_bitmap, meters2pixels(LINK_HEIGHT/2 + 2*LINK_LENGTH/(3.5)), meters2pixels(LINK_HEIGHT/2), meters2pixels(LINK_HEIGHT/6), border_color);
        circlefill(link_bitmap, meters2pixels(LINK_HEIGHT/2 + 2*LINK_LENGTH/(3.5)), meters2pixels(LINK_HEIGHT/2), meters2pixels(LINK_HEIGHT/6) - border_size, 0);
}
//------------------------------------------------------------------------------------------------------------

void draw_beam_bitmap(BITMAP* beam_bitmap)
{

int     border_size = 2;                // Size in pixels of black contour of drawing components
int     border_color = 16;              // ALmost black
int     beam_color = 28;                // Type of gray

        // Beam:
        rectfill(beam_bitmap, meters2pixels(LASER_L1 + LASER_L2), meters2pixels(BEAM_HEIGHT - BEAM_THICKNESS), 
                              beam_bitmap->w, beam_bitmap->h, border_color);
        rectfill(beam_bitmap, (meters2pixels(LASER_L1 + LASER_L2) + border_size), (meters2pixels(BEAM_HEIGHT - BEAM_THICKNESS) + border_size), 
                              (beam_bitmap->w - border_size - 1), (beam_bitmap->h - border_size - 1), beam_color);        
        // Left Beam Limit:
        rectfill(beam_bitmap, meters2pixels(LASER_L1 + LASER_L2), meters2pixels(BEAM_HEIGHT/2), 
                              meters2pixels(LASER_L1 + LASER_L2) + meters2pixels(BEAM_THICKNESS_2), beam_bitmap->h, border_color);
        rectfill(beam_bitmap, (meters2pixels(LASER_L1 + LASER_L2) + border_size), (meters2pixels(BEAM_HEIGHT/2)+ border_size), 
                               meters2pixels(LASER_L1 + LASER_L2) + (meters2pixels(BEAM_THICKNESS_2) - border_size), (beam_bitmap->h - border_size), beam_color);
        // Right Beam Limit:
        rectfill(beam_bitmap, meters2pixels(LASER_L1 + LASER_L2 + BEAM_THICKNESS_2 + BEAM_LENGTH), 0, 
                              beam_bitmap->w, beam_bitmap->h, border_color);
        rectfill(beam_bitmap, (meters2pixels(LASER_L1 + LASER_L2 + BEAM_THICKNESS_2 + BEAM_LENGTH) + border_size), (0 + border_size), 
                              (beam_bitmap->w - border_size - 1), (beam_bitmap->h - border_size - 1), beam_color);
}
//------------------------------------------------------------------------------------------------------------

void draw_beam_support(BITMAP* system_bitmap)
{

int     border_size = 2;                // Size in pixels of black contour of drawing components
int     border_color = 16;                
int     support_color = 20;
int     hinge_color = 28;     

        // Vertical Support:
        rectfill(system_bitmap, world2pixel_x_coordinate(PIVOT_X - (float)SUPPORT_THICKNESS/2), world2pixel_y_coordinate(PIVOT_Y),
                 world2pixel_x_coordinate(L1 + (float)BASE_WIDTH/2 + (float)SUPPORT_THICKNESS/2), world2pixel_y_coordinate(0),
                 border_color);
        rectfill(system_bitmap, world2pixel_x_coordinate(PIVOT_X - (float)SUPPORT_THICKNESS/2) + border_size, world2pixel_y_coordinate(PIVOT_Y) + border_size,
                 world2pixel_x_coordinate(L1 + (float)BASE_WIDTH/2 + (float)SUPPORT_THICKNESS/2) - border_size, world2pixel_y_coordinate(0) - border_size,
                 support_color);         
        circlefill(system_bitmap, world2pixel_x_coordinate(PIVOT_X), world2pixel_y_coordinate(PIVOT_Y), meters2pixels((float)SUPPORT_THICKNESS/2), border_color);
        circlefill(system_bitmap, world2pixel_x_coordinate(PIVOT_X), world2pixel_y_coordinate(PIVOT_Y), meters2pixels((float)SUPPORT_THICKNESS/2) - border_size, support_color);                 
        // Base Support:
        rectfill(system_bitmap, world2pixel_x_coordinate(L1), world2pixel_y_coordinate(BASE_HEIGHT),
                 world2pixel_x_coordinate(L1 + BASE_WIDTH), world2pixel_y_coordinate(0),
                 border_color);
        rectfill(system_bitmap, world2pixel_x_coordinate(L1) + border_size, world2pixel_y_coordinate(BASE_HEIGHT) + border_size,
                 world2pixel_x_coordinate(L1 + BASE_WIDTH) - border_size, world2pixel_y_coordinate(0) - border_size,
                 support_color);                                                  
        // Hinge:
        circlefill(system_bitmap, world2pixel_x_coordinate(PIVOT_X), world2pixel_y_coordinate(PIVOT_Y), meters2pixels(0.01), border_color);
        circlefill(system_bitmap, world2pixel_x_coordinate(PIVOT_X), world2pixel_y_coordinate(PIVOT_Y), meters2pixels(0.01) - border_size, hinge_color);
}
//------------------------------------------------------------------------------------------------------------

void draw_link_2(float theta_beam, float theta_motor, BITMAP* system_bitmap)
{

int     support_color = 20;        
int     P1_x, P1_y, P2_x, P2_y, P3_x, P3_y, P4_x, P4_y;         // Polygon 4 coordinates

        P1_x = world2pixel_x_coordinate(MOTOR_CENTER_X + (LINK_LENGTH - LINK_D1 - LINK_D1/3 - LINK_HEIGHT/2)*cos(theta_motor));
        P1_y = world2pixel_y_coordinate(MOTOR_CENTER_Y + (LINK_LENGTH - LINK_D1 - LINK_D1/3 - LINK_HEIGHT/2)*sin(theta_motor));
        P2_x = world2pixel_x_coordinate(MOTOR_CENTER_X + (LINK_LENGTH - LINK_D1 + LINK_D1/3 - LINK_HEIGHT/2)*cos(theta_motor));
        P2_y = world2pixel_y_coordinate(MOTOR_CENTER_Y + (LINK_LENGTH - LINK_D1 + LINK_D1/3 - LINK_HEIGHT/2)*sin(theta_motor));
        P3_x = world2pixel_x_coordinate(PIVOT_X + BEAM_THICKNESS/2*sin(theta_beam) - PIVOT_DISTANCE*cos(theta_beam) + BEAM_LENGTH*cos(theta_beam) + LINK_D1*sin(theta_beam) + LINK_D1/3*cos(theta_beam));
        P3_y = world2pixel_y_coordinate(PIVOT_Y - BEAM_THICKNESS/2*cos(theta_beam) - PIVOT_DISTANCE*sin(theta_beam) + BEAM_LENGTH*sin(theta_beam) - LINK_D1*cos(theta_beam) + LINK_D1/3*sin(theta_beam));
        P4_x = world2pixel_x_coordinate(PIVOT_X + BEAM_THICKNESS/2*sin(theta_beam) - PIVOT_DISTANCE*cos(theta_beam) + BEAM_LENGTH*cos(theta_beam) + LINK_D1*sin(theta_beam) - LINK_D1/3*cos(theta_beam));
        P4_y = world2pixel_y_coordinate(PIVOT_Y - BEAM_THICKNESS/2*cos(theta_beam) - PIVOT_DISTANCE*sin(theta_beam) + BEAM_LENGTH*sin(theta_beam) - LINK_D1*cos(theta_beam) - LINK_D1/3*sin(theta_beam));

int     polygon_2[8] = {P1_x, P1_y, P2_x, P2_y, P3_x, P3_y, P4_x, P4_y};

        polygon(system_bitmap, 4, polygon_2, support_color);
}
//------------------------------------------------------------------------------------------------------------

void draw_beam_support_2(float theta_beam, BITMAP* system_bitmap)
{

int     beam_support_2_color = 20;                                              // Type of gray  
int     P11_x, P11_y, P12_x, P12_y, P13_x, P13_y, P14_x, P14_y;                 // Polygon coordinates
float   dist_1 = BEAM_THICKNESS_2;
float   eps = (float)1/553;                                                     // other little distance for the drawing [m] 
 
        P11_x = world2pixel_x_coordinate(PIVOT_X + (BEAM_THICKNESS - eps)/2*sin(theta_beam) - PIVOT_DISTANCE*cos(theta_beam) + BEAM_LENGTH*cos(theta_beam)-BEAM_THICKNESS_2*cos(theta_beam));
        P11_y = world2pixel_y_coordinate(PIVOT_Y - (BEAM_THICKNESS - eps)/2*cos(theta_beam) - PIVOT_DISTANCE*sin(theta_beam) + BEAM_LENGTH*sin(theta_beam)-BEAM_THICKNESS_2*sin(theta_beam));
        P12_x = world2pixel_x_coordinate(PIVOT_X + (BEAM_THICKNESS - eps)/2*sin(theta_beam) - PIVOT_DISTANCE*cos(theta_beam) + BEAM_LENGTH*cos(theta_beam)-BEAM_THICKNESS_2*cos(theta_beam) + dist_1*sin(theta_beam));
        P12_y = world2pixel_y_coordinate(PIVOT_Y - (BEAM_THICKNESS - eps)/2*cos(theta_beam) - PIVOT_DISTANCE*sin(theta_beam) + BEAM_LENGTH*sin(theta_beam)-BEAM_THICKNESS_2*sin(theta_beam) - dist_1*cos(theta_beam));
        P13_x = world2pixel_x_coordinate(PIVOT_X + (BEAM_THICKNESS - eps)/2*sin(theta_beam) - PIVOT_DISTANCE*cos(theta_beam) + BEAM_LENGTH*cos(theta_beam)+BEAM_THICKNESS_2*cos(theta_beam) + dist_1*sin(theta_beam));
        P13_y = world2pixel_y_coordinate(PIVOT_Y - (BEAM_THICKNESS - eps)/2*cos(theta_beam) - PIVOT_DISTANCE*sin(theta_beam) + BEAM_LENGTH*sin(theta_beam)+BEAM_THICKNESS_2*sin(theta_beam) - dist_1*cos(theta_beam));
        P14_x = world2pixel_x_coordinate(PIVOT_X + (BEAM_THICKNESS - eps)/2*sin(theta_beam) - PIVOT_DISTANCE*cos(theta_beam) + BEAM_LENGTH*cos(theta_beam)+BEAM_THICKNESS_2*cos(theta_beam));
        P14_y = world2pixel_y_coordinate(PIVOT_Y - (BEAM_THICKNESS - eps)/2*cos(theta_beam) - PIVOT_DISTANCE*sin(theta_beam) + BEAM_LENGTH*sin(theta_beam)+BEAM_THICKNESS_2*sin(theta_beam));

int     polygon_1[8] = {P11_x, P11_y, P12_x, P12_y, P13_x, P13_y, P14_x, P14_y};

        polygon(system_bitmap, 4, polygon_1, beam_support_2_color);
        // Contour:
        line(system_bitmap, P11_x, P11_y, P12_x, P12_y, 16);
        //line(screen, P11_x, P11_y, P14_x, P14_y, 16);
        line(system_bitmap, P12_x, P12_y, P13_x, P13_y, 16);
        line(system_bitmap, P13_x, P13_y, P14_x, P14_y, 16);
}
//------------------------------------------------------------------------------------------------------------
void draw_Q_table_legend()
{

int     space = 6;             
char    s_2[] = "LEGEND:";
int     legenda_start_y = UPPER_LEFT_Q_BOX_Y + 35; 
int     legenda_cell_size = 13;
int     border_size = 2;
        textout_ex(screen, font, s_2, UPPER_LEFT_Q_BOX_X + space, legenda_start_y, 16, -1);

        rectfill(screen, UPPER_LEFT_Q_BOX_X + space, legenda_start_y + legenda_cell_size + space, UPPER_LEFT_Q_BOX_X + space + legenda_cell_size, legenda_start_y + legenda_cell_size + space + legenda_cell_size, 16);
        rectfill(screen, UPPER_LEFT_Q_BOX_X + space + border_size, legenda_start_y + legenda_cell_size + space + border_size, UPPER_LEFT_Q_BOX_X + space + legenda_cell_size - border_size, legenda_start_y + legenda_cell_size + space + legenda_cell_size - border_size, makecol8(150, 0, 0));
        textout_ex(screen, font, "Q<-50", UPPER_LEFT_Q_BOX_X + 2*space + legenda_cell_size, legenda_start_y + legenda_cell_size + space + 4, 16, -1);

        rectfill(screen, UPPER_LEFT_Q_BOX_X + space, legenda_start_y + 2*legenda_cell_size + 2*space, UPPER_LEFT_Q_BOX_X + space + legenda_cell_size, legenda_start_y + 2*legenda_cell_size + 2*space + legenda_cell_size, 16);
        rectfill(screen, UPPER_LEFT_Q_BOX_X + space + border_size, legenda_start_y + 2*legenda_cell_size + 2*space + border_size, UPPER_LEFT_Q_BOX_X + space + legenda_cell_size - border_size, legenda_start_y + legenda_cell_size + 2*space + 2*legenda_cell_size - border_size, makecol8(255, 0, 0));
        textout_ex(screen, font, "-50<=Q<-30", UPPER_LEFT_Q_BOX_X + 2*space + legenda_cell_size, legenda_start_y + 2*legenda_cell_size + 2*space + 4, 16, -1);

        rectfill(screen, UPPER_LEFT_Q_BOX_X + space, legenda_start_y + 3*legenda_cell_size + 3*space, UPPER_LEFT_Q_BOX_X + space + legenda_cell_size, legenda_start_y + 3*legenda_cell_size + 3*space + legenda_cell_size, 16);
        rectfill(screen, UPPER_LEFT_Q_BOX_X + space + border_size, legenda_start_y + 3*legenda_cell_size + 3*space + border_size, UPPER_LEFT_Q_BOX_X + space + legenda_cell_size - border_size, legenda_start_y + legenda_cell_size + 3*space + 3*legenda_cell_size - border_size, makecol8(255, 165, 0));
        textout_ex(screen, font, "-30<=Q<-15", UPPER_LEFT_Q_BOX_X + 2*space + legenda_cell_size, legenda_start_y + 3*legenda_cell_size + 3*space + 4, 16, -1);

        rectfill(screen, UPPER_LEFT_Q_BOX_X + space, legenda_start_y + 4*legenda_cell_size + 4*space, UPPER_LEFT_Q_BOX_X + space + legenda_cell_size, legenda_start_y + 4*legenda_cell_size + 4*space + legenda_cell_size, 16);
        rectfill(screen, UPPER_LEFT_Q_BOX_X + space + border_size, legenda_start_y + 4*legenda_cell_size + 4*space + border_size, UPPER_LEFT_Q_BOX_X + space + legenda_cell_size - border_size, legenda_start_y + legenda_cell_size + 4*space + 4*legenda_cell_size - border_size, makecol8(255, 255, 0));
        textout_ex(screen, font, "-15<=Q<0", UPPER_LEFT_Q_BOX_X + 2*space + legenda_cell_size, legenda_start_y + 4*legenda_cell_size + 4*space + 4, 16, -1);

        rectfill(screen, UPPER_LEFT_Q_BOX_X + space, legenda_start_y + 5*legenda_cell_size + 5*space, UPPER_LEFT_Q_BOX_X + space + legenda_cell_size, legenda_start_y + 5*legenda_cell_size + 5*space + legenda_cell_size, 16);
        rectfill(screen, UPPER_LEFT_Q_BOX_X + space + border_size, legenda_start_y + 5*legenda_cell_size + 5*space + border_size, UPPER_LEFT_Q_BOX_X + space + legenda_cell_size - border_size, legenda_start_y + legenda_cell_size + 5*space + 5*legenda_cell_size - border_size, makecol8(255, 255, 255));
        textout_ex(screen, font, "Q=0", UPPER_LEFT_Q_BOX_X + 2*space + legenda_cell_size, legenda_start_y + 5*legenda_cell_size + 5*space + 4, 16, -1);

        rectfill(screen, UPPER_LEFT_Q_BOX_X + space, legenda_start_y + 6*legenda_cell_size + 6*space, UPPER_LEFT_Q_BOX_X + space + legenda_cell_size, legenda_start_y + 6*legenda_cell_size + 6*space + legenda_cell_size, 16);
        rectfill(screen, UPPER_LEFT_Q_BOX_X + space + border_size, legenda_start_y + 6*legenda_cell_size + 6*space + border_size, UPPER_LEFT_Q_BOX_X + space + legenda_cell_size - border_size, legenda_start_y + legenda_cell_size + 6*space + 6*legenda_cell_size - border_size, makecol8(144, 238, 144));
        textout_ex(screen, font, "0<Q<=15", UPPER_LEFT_Q_BOX_X + 2*space + legenda_cell_size, legenda_start_y + 6*legenda_cell_size + 6*space + 4, 16, -1);

        rectfill(screen, UPPER_LEFT_Q_BOX_X + space, legenda_start_y + 7*legenda_cell_size + 7*space, UPPER_LEFT_Q_BOX_X + space + legenda_cell_size, legenda_start_y + 7*legenda_cell_size + 7*space + legenda_cell_size, 16);
        rectfill(screen, UPPER_LEFT_Q_BOX_X + space + border_size, legenda_start_y + 7*legenda_cell_size + 7*space + border_size, UPPER_LEFT_Q_BOX_X + space + legenda_cell_size - border_size, legenda_start_y + legenda_cell_size + 7*space + 7*legenda_cell_size - border_size, makecol8(0, 255, 0));
        textout_ex(screen, font, "15<Q<=30", UPPER_LEFT_Q_BOX_X + 2*space + legenda_cell_size, legenda_start_y + 7*legenda_cell_size + 7*space + 4, 16, -1);

        rectfill(screen, UPPER_LEFT_Q_BOX_X + space, legenda_start_y + 8*legenda_cell_size + 8*space, UPPER_LEFT_Q_BOX_X + space + legenda_cell_size, legenda_start_y + 8*legenda_cell_size + 8*space + legenda_cell_size, 16);
        rectfill(screen, UPPER_LEFT_Q_BOX_X + space + border_size, legenda_start_y + 8*legenda_cell_size + 8*space + border_size, UPPER_LEFT_Q_BOX_X + space + legenda_cell_size - border_size, legenda_start_y + legenda_cell_size + 8*space + 8*legenda_cell_size - border_size, makecol8(50, 205, 50));
        textout_ex(screen, font, "30<Q<=50", UPPER_LEFT_Q_BOX_X + 2*space + legenda_cell_size, legenda_start_y + 8*legenda_cell_size + 8*space + 4, 16, -1);

        rectfill(screen, UPPER_LEFT_Q_BOX_X + space, legenda_start_y + 9*legenda_cell_size + 9*space, UPPER_LEFT_Q_BOX_X + space + legenda_cell_size, legenda_start_y + 9*legenda_cell_size + 9*space + legenda_cell_size, 16);
        rectfill(screen, UPPER_LEFT_Q_BOX_X + space + border_size, legenda_start_y + 9*legenda_cell_size + 9*space + border_size, UPPER_LEFT_Q_BOX_X + space + legenda_cell_size - border_size, legenda_start_y + legenda_cell_size + 9*space + 9*legenda_cell_size - border_size, makecol8(0, 100, 0));
        textout_ex(screen, font, "Q>50", UPPER_LEFT_Q_BOX_X + 2*space + legenda_cell_size, legenda_start_y + 9*legenda_cell_size + 9*space + 4, 16, -1);

        rectfill(screen, UPPER_LEFT_Q_BOX_X + space, legenda_start_y + 10*legenda_cell_size + 10*space, UPPER_LEFT_Q_BOX_X + space + legenda_cell_size, legenda_start_y + 10*legenda_cell_size + 10*space + legenda_cell_size, 16);
        rectfill(screen, UPPER_LEFT_Q_BOX_X + space + border_size, legenda_start_y + 10*legenda_cell_size + 10*space + border_size, UPPER_LEFT_Q_BOX_X + space + legenda_cell_size - border_size, legenda_start_y + legenda_cell_size + 10*space + 10*legenda_cell_size - border_size, 32);
        textout_ex(screen, font, "CURRENT STATUS", UPPER_LEFT_Q_BOX_X + 2*space + legenda_cell_size, legenda_start_y + 10*legenda_cell_size + 10*space + 4, 16, -1);
}
//------------------------------------------------------------------------------------------------------------

void draw_Q_table_button_instructions()
{
int     space = 6;                              // Some pixel spacing
int     space_2 = 45;                           // Distance between buttons instructions
int     instructions_start_y =   260;
int     instructions_start_y_2 = instructions_start_y + space_2;
int     instructions_start_y_3 = instructions_start_y + 2*space_2;
int     instructions_start_y_4 = instructions_start_y + 3*space_2;
int     instructions_start_y_5 = instructions_start_y + 4*space_2;
int     instructions_start_y_6 = instructions_start_y + 5*space_2;
int     instructions_start_y_7 = instructions_start_y + 6*space_2;
int     instructions_start_y_8 = instructions_start_y + 7*space_2;

        textout_ex(screen, font, "PRESS:", UPPER_LEFT_Q_BOX_X + space, instructions_start_y, 16, -1);
        draw_button_key(UPPER_LEFT_Q_BOX_X + space + text_length(font, "PRESS:") + 10, instructions_start_y - 5, "S");
        textout_ex(screen, font, "TO SAVE Q-MATRIX", UPPER_LEFT_Q_BOX_X + space, instructions_start_y + text_height(font) + space + 5, 16, -1);

        textout_ex(screen, font, "PRESS:", UPPER_LEFT_Q_BOX_X + space, instructions_start_y_2, 16, -1);
        draw_button_key(UPPER_LEFT_Q_BOX_X + space + text_length(font, "PRESS:") + 10, instructions_start_y_2 - 5, "L");
        textout_ex(screen, font, "TO LOAD Q-MATRIX", UPPER_LEFT_Q_BOX_X + space, instructions_start_y_2 + text_height(font) + space + 5, 16, -1);

        textout_ex(screen, font, "PRESS:", UPPER_LEFT_Q_BOX_X + space, instructions_start_y_3, 16, -1);
        draw_button_key(UPPER_LEFT_Q_BOX_X + space + text_length(font, "PRESS:") + 10, instructions_start_y_3 - 5, "F");
        textout_ex(screen, font, "TO FAST MODE", UPPER_LEFT_Q_BOX_X + space, instructions_start_y_3 + text_height(font) + space + 5, 16, -1);

        textout_ex(screen, font, "PRESS:", UPPER_LEFT_Q_BOX_X + space, instructions_start_y_4, 16, -1);
        draw_button_key(UPPER_LEFT_Q_BOX_X + space + text_length(font, "PRESS:") + 10, instructions_start_y_4 - 5, "N");
        textout_ex(screen, font, "TO NORMAL MODE", UPPER_LEFT_Q_BOX_X + space, instructions_start_y_4 + text_height(font) + space + 5, 16, -1); 
 
        textout_ex(screen, font, "PRESS:", UPPER_LEFT_Q_BOX_X + space, instructions_start_y_5, 16, -1);
        draw_button_key(UPPER_LEFT_Q_BOX_X + space + text_length(font, "PRESS:") + 10, instructions_start_y_5 - 5, "+");
        textout_ex(screen, font, "TO INCREASE FRICTION", UPPER_LEFT_Q_BOX_X + space, instructions_start_y_5 + text_height(font) + space + 5, 16, -1);

        textout_ex(screen, font, "PRESS:", UPPER_LEFT_Q_BOX_X + space, instructions_start_y_6, 16, -1);
        draw_button_key(UPPER_LEFT_Q_BOX_X + space + text_length(font, "PRESS:") + 10, instructions_start_y_6 - 5, "-");
        textout_ex(screen, font, "TO DECREASE FRICTION", UPPER_LEFT_Q_BOX_X + space, instructions_start_y_6 + text_height(font) + space + 5, 16, -1);

        textout_ex(screen, font, "PRESS:", UPPER_LEFT_Q_BOX_X + space, instructions_start_y_7, 16, -1);
        draw_button_key(UPPER_LEFT_Q_BOX_X + space + text_length(font, "PRESS:") + 10, instructions_start_y_7 - 5, "B");
        textout_ex(screen, font, "TO BOLTZMANN POLICY", UPPER_LEFT_Q_BOX_X + space, instructions_start_y_7 + text_height(font) + space + 5, 16, -1);

        textout_ex(screen, font, "PRESS:", UPPER_LEFT_Q_BOX_X + space, instructions_start_y_8, 16, -1);
        draw_button_key(UPPER_LEFT_Q_BOX_X + space + text_length(font, "PRESS:") + 10, instructions_start_y_8 - 5, "G");
        textout_ex(screen, font, "TO EPS-GREEDY POLICY", UPPER_LEFT_Q_BOX_X + space, instructions_start_y_8 + text_height(font) + space + 5, 16, -1);
}
//------------------------------------------------------------------------------------------------------------

void draw_empty_TD_err_plot()
{
int     i;                                                                      // Iterator
int     grid_color = 30;
int     grid_spacing = 10;                                                      // size in pixels of the grid gray cell
int     vert_grid_line_num = floor((float)TD_ERR_PLOT_WIDTH/grid_spacing);
int     horiz_grid_line_num = floor((float)TD_ERR_PLOT_HEIGHT/grid_spacing);
int     x_values_spacing = 5*grid_spacing;                                      // space between x value texts in the plot
int     y_values_spacing = 3*grid_spacing;                                      // space between y value texts in the plot
int     x_values_num = floor((float)TD_ERR_PLOT_WIDTH/x_values_spacing);
int     y_values_num = floor((float)TD_ERR_PLOT_HEIGHT/y_values_spacing);
int     epi_num_interval = 50;
int     td_err_interval = 15;
char    s[4];           // string buffer

// Title:
        rectfill(screen, UPPER_LEFT_TD_ERR_PLOT_X , UPPER_LEFT_TD_ERR_PLOT_Y - 2*text_height(font) - 2, UPPER_LEFT_TD_ERR_PLOT_X + 150, UPPER_LEFT_TD_ERR_PLOT_Y - 6, 16);
        rectfill(screen, UPPER_LEFT_TD_ERR_PLOT_X + 2, UPPER_LEFT_TD_ERR_PLOT_Y - 2*text_height(font) - 2 + 2, UPPER_LEFT_TD_ERR_PLOT_X + 150 + 2, UPPER_LEFT_TD_ERR_PLOT_Y - 6 + 2, 16);
        rectfill(screen, UPPER_LEFT_TD_ERR_PLOT_X + 1, UPPER_LEFT_TD_ERR_PLOT_Y - 2*text_height(font) - 2 + 1, UPPER_LEFT_TD_ERR_PLOT_X + 150 - 1, UPPER_LEFT_TD_ERR_PLOT_Y - 6 - 1, makecol8(255,255,255));
        textout_ex(screen, font, "TD-ERR VS EPISODE:", UPPER_LEFT_TD_ERR_PLOT_X + 1 + 4, UPPER_LEFT_TD_ERR_PLOT_Y - 2*text_height(font) - 2 + 1 + 2, 16, -1);

// Gray Grid Drawing:
        for (i=1; i <= vert_grid_line_num; i++) {
                line(screen, UPPER_LEFT_TD_ERR_PLOT_X + i*grid_spacing, UPPER_LEFT_TD_ERR_PLOT_Y, UPPER_LEFT_TD_ERR_PLOT_X + i*grid_spacing, LOWER_RIGHT_TD_ERR_PLOT_Y, grid_color);
        }
        for (i=1; i < horiz_grid_line_num; i++) {
                line(screen, UPPER_LEFT_TD_ERR_PLOT_X, UPPER_LEFT_TD_ERR_PLOT_Y + i*grid_spacing, LOWER_RIGHT_TD_ERR_PLOT_X, UPPER_LEFT_TD_ERR_PLOT_Y + i*grid_spacing, grid_color);
        }
        textout_ex(screen, font, "0", UPPER_LEFT_TD_ERR_PLOT_X + 2, LOWER_RIGHT_TD_ERR_PLOT_Y - text_height(font), 25, -1);
// X-Y values drawing:
        for (i=1; i <= x_values_num; i++) {
                line(screen, UPPER_LEFT_TD_ERR_PLOT_X + i*x_values_spacing, LOWER_RIGHT_TD_ERR_PLOT_Y, UPPER_LEFT_TD_ERR_PLOT_X + i*x_values_spacing, LOWER_RIGHT_TD_ERR_PLOT_Y - grid_spacing, 16);
                sprintf(s, "%d", i*epi_num_interval);
                textout_ex(screen, font, s, UPPER_LEFT_TD_ERR_PLOT_X + i*x_values_spacing + 2, LOWER_RIGHT_TD_ERR_PLOT_Y - text_height(font), 25, -1);
        }

        for (i=1; i < y_values_num; i++) {
                line(screen, UPPER_LEFT_TD_ERR_PLOT_X, LOWER_RIGHT_TD_ERR_PLOT_Y - i*y_values_spacing, UPPER_LEFT_TD_ERR_PLOT_X + grid_spacing, LOWER_RIGHT_TD_ERR_PLOT_Y - i*y_values_spacing, 16);
                sprintf(s, "%d", i*td_err_interval);
                textout_ex(screen, font, s, UPPER_LEFT_TD_ERR_PLOT_X + grid_spacing + 2, LOWER_RIGHT_TD_ERR_PLOT_Y - i*y_values_spacing - text_height(font)/2, 25, -1);

        }
}
//------------------------------------------------------------------------------------------------------------

int  td_err_2_y_plot(float td_error)
{
float   td_error_max = 60.0;    // Per ora e' ancora in fase di sviluppo.. non so mi serviranno valori piu alti!
int     pixel_y;                // y pixel coordinate of the td_error value

        pixel_y = td_error/td_error_max*TD_ERR_PLOT_HEIGHT;
        pixel_y = LOWER_RIGHT_TD_ERR_PLOT_Y - pixel_y; 

        return pixel_y;
}
//------------------------------------------------------------------------------------------------------------

void update_TD_err_plot()
{

int     line_color = makecol8(255,0,0);
int     grid_spacing = 10;      // size in pixels of the grid gray cell
int     pixel_x, pixel_y;

        // After the first episode draw the relative pixel:
        if (rl_episode_counter == 2 && td_plot_flag) {

                pixel_x = UPPER_LEFT_TD_ERR_PLOT_X;
                pixel_y = td_err_2_y_plot(avg_episode_TD_error);
                if (pixel_y <= UPPER_LEFT_TD_ERR_PLOT_Y) {
                        pixel_y = UPPER_LEFT_TD_ERR_PLOT_Y;
                }
                putpixel(screen, pixel_x, pixel_y, line_color);
                td_plot_x_pixel_old = pixel_x;
                td_plot_y_pixel_old = pixel_y;
                td_plot_flag = 0;
        }

        // Then draw the td error only every 10 episodes:
        if (rl_episode_counter % 10 == 0 && td_plot_flag) {
                td_plot_step++;

                pixel_x = UPPER_LEFT_TD_ERR_PLOT_X + td_plot_step*grid_spacing;
                pixel_y = td_err_2_y_plot(avg_episode_TD_error);
                if (pixel_y <= UPPER_LEFT_TD_ERR_PLOT_Y) pixel_y = UPPER_LEFT_TD_ERR_PLOT_Y;

                if (pixel_x <= LOWER_RIGHT_TD_ERR_PLOT_X) {
                        line(screen, td_plot_x_pixel_old, td_plot_y_pixel_old, pixel_x, pixel_y, line_color);
                }
                else printf("The TD-error won't be updated anymore. Maximum plot size reached!\n");
                td_plot_x_pixel_old = pixel_x;
                td_plot_y_pixel_old = pixel_y;
                td_plot_flag = 0;
        }
}
//------------------------------------------------------------------------------------------------------------

void set_avg_episode_TD_err(float td_err)
{
        avg_episode_TD_error = td_err;
        td_plot_flag = 1;
}
//------------------------------------------------------------------------------------------------------------
