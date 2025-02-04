#define _POSIX_C_SOURCE 200112L
//============================================================================================================
// USER AND COMMAND INTERPRETER LIBRARY HEADER
//============================================================================================================

//------------------------------------------------------------------------------------------------------------
// USER TASK CONSTANTS
//------------------------------------------------------------------------------------------------------------
#define COMMAND_PRIO   85    // task priority
#define COMMAND_DL     30    // relative deadline in [ms]
#define COMMAND_PER    30    // period in [ms]

//------------------------------------------------------------------------------------------------------------
// USER FUNCTION PROTOTYPES
//------------------------------------------------------------------------------------------------------------

/*
This function get the keycodes of the pressed key and write them at the given pointers
*/
void get_key_codes(char* scan, char* ascii);

/*
This function performs actions based on the keyboard commands
*/
void command_interpreter();

/*
This function defines the user_task
*/
void* user_task(void* arg);

/*
This function do the initialization, install the keyboard, and starts the user_task
*/
void user_init();
