# Makefile for RTL1 Volume Solver

CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -fopenmp

# Volume algorithm source path (escape spaces)
VOL_BASE = /home/local.isima.fr/antran/UFF/QAP_New_formulation
VOL_PATH = $(VOL_BASE)/VolQAP\ copy\ 2
VOL_INCDIR = $(VOL_PATH)/includes
VOL_SRCDIR = $(VOL_PATH)/src

# Include flags
INCFLAGS = -I$(VOL_INCDIR) -I../../include

# Source files
SRC = rtl1_volume_solver.cpp
OBJ = rtl1_volume_solver.o VolVolume.o
EXE = rtl1_volume_solver

# Build rules
all: $(EXE)

$(EXE): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lm -lpthread

rtl1_volume_solver.o: $(SRC)
	$(CXX) $(CXXFLAGS) $(INCFLAGS) -c -o $@ $<

VolVolume.o: $(VOL_SRCDIR)/VolVolume.cpp
	$(CXX) $(CXXFLAGS) -I$(VOL_INCDIR) -c -o $@ "$<"

clean:
	rm -f $(OBJ) $(EXE)

.PHONY: all clean
