#---------------------------------------------------
# Directories
#---------------------------------------------------
SRC_DIR = ./src/
LIB_DIR = ./libs/
BUILD_DIR = ./build/
#---------------------------------------------------
# Target file to be compiled by default
#---------------------------------------------------
MAIN = ball_and_beam_by_RL_main
#---------------------------------------------------
# CC is the compiler to be used
#---------------------------------------------------
CC = gcc
#---------------------------------------------------
# CFLAGS are the options passed to the compiler
#---------------------------------------------------
CFLAGS = -Wall -Ilibs
#---------------------------------------------------
# OBJS are the object files to be linked
#---------------------------------------------------
OBJ1 = ptask
OBJ2 = ball_and_beam_dynamics
OBJ3 = my_graphics
OBJ4 = command_interpreter
OBJ5 = qlearn

OBJS = $(BUILD_DIR)$(MAIN).o $(BUILD_DIR)$(OBJ1).o $(BUILD_DIR)$(OBJ2).o $(BUILD_DIR)$(OBJ3).o $(BUILD_DIR)$(OBJ4).o $(BUILD_DIR)$(OBJ5).o
#---------------------------------------------------
# LIBS are the external libraries to be used
#---------------------------------------------------
LIBS = -lpthread -lrt -lm `allegro-config --libs`

#---------------------------------------------------
# Dependencies
#---------------------------------------------------

$(BUILD_DIR)$(MAIN): $(OBJS)
	$(CC) -o $(BUILD_DIR)$(MAIN) $(OBJS) $(LIBS)

$(BUILD_DIR)$(MAIN).o: $(SRC_DIR)$(MAIN).c
	$(CC) $(CFLAGS) -c $(SRC_DIR)$(MAIN).c -o $(BUILD_DIR)$(MAIN).o

$(BUILD_DIR)$(OBJ1).o: $(LIB_DIR)$(OBJ1).c
	$(CC) $(CFLAGS) -c $(LIB_DIR)$(OBJ1).c -o $(BUILD_DIR)$(OBJ1).o			

$(BUILD_DIR)$(OBJ2).o: $(LIB_DIR)$(OBJ2).c
	$(CC) $(CFLAGS) -c $(LIB_DIR)$(OBJ2).c -o $(BUILD_DIR)$(OBJ2).o			

$(BUILD_DIR)$(OBJ3).o: $(LIB_DIR)$(OBJ3).c
	$(CC) $(CFLAGS) -c $(LIB_DIR)$(OBJ3).c -o $(BUILD_DIR)$(OBJ3).o			

$(BUILD_DIR)$(OBJ4).o: $(LIB_DIR)$(OBJ4).c
	$(CC) $(CFLAGS) -c $(LIB_DIR)$(OBJ4).c -o $(BUILD_DIR)$(OBJ4).o	

$(BUILD_DIR)$(OBJ5).o: $(LIB_DIR)$(OBJ5).c
	$(CC) $(CFLAGS) -c $(LIB_DIR)$(OBJ5).c -o $(BUILD_DIR)$(OBJ5).o								

#---------------------------------------------------
# Command to delete object and executable files
# It can be specified inline: make clean
# The clean is executed only if you invoke "make clean" 
#---------------------------------------------------
clean:
	rm -rf $(BUILD_DIR)/*.o $(BUILD_DIR)/$(MAIN)