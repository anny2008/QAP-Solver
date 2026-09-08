# Makefile for sdp Volume Solver

CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -Wall -Wextra -fopenmp -fPIC



CPLEX_PATH = /opt/ibm/ILOG/CPLEX_Studio2211
GUROBI_HOME = /opt/gurobi/gurobi1200/linux64
EIGEN_INC_PATH = /home/local.isima.fr/antran/.local/include
MOSEK_PATH = /home/local.isima.fr/antran/mosek/mosek/11.2/tools/platform/linux64x86
# ---- Include Paths ----
# We are in cpp/modules/cg_cplex, so go up two levels to reach core/include
INCLUDES = \
    -I$(CPLEX_PATH)/concert/include \
    -I$(CPLEX_PATH)/cplex/include \
    -I$(EIGEN_INC_PATH) \
    -I$(MOSEK_PATH)/h \
    -I../../core \
    -I../../include \
    -I.

# ---- Library Paths (static PIC recommended) ----
CONCERT_LIB = $(CPLEX_PATH)/concert/lib/x86-64_linux/static_pic
CPLEX_LIB   = $(CPLEX_PATH)/cplex/lib/x86-64_linux/static_pic

LIBS = \
    $(CONCERT_LIB)/libconcert.a \
    $(CPLEX_LIB)/libilocplex.a \
    $(CPLEX_LIB)/libcplex.a \
    -L$(MOSEK_PATH)/bin \
    -lfusion64 -lmosek64 \
    -lpthread -lm -ldl

SRC = \
    sdp_volume_solver.cpp \
    VolVolume.cpp \
    Hungarian.cpp \
    ADMM.cpp \
    main.cpp

OBJ = $(SRC:.cpp=.o)

# ---- Output executable ----
TARGET = sdp_volume_solver

# ============================================
# Build Rules
# ============================================

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean