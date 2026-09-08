# ============================================
# Makefile for rlt1_cg (local to this folder)
# Directory: cpp/modules/rlt1_cg
# Builds:   rlt1_cg
# ============================================

# ---- Compiler ----
CXX      = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -fPIC
CPLEX_PATH = /opt/ibm/ILOG/CPLEX_Studio2211
GUROBI_HOME = /opt/gurobi/gurobi1200/linux64
# ---- Include Paths ----
# We are in cpp/modules/rlt1_cg, so go up two levels to reach core/include
INCLUDES = \
    -I$(GUROBI_HOME)/include \
    -I$(CPLEX_PATH)/concert/include \
    -I$(CPLEX_PATH)/cplex/include \
    -I../../core \
    -I../../include \
    -I.

# ---- Library Paths (static PIC recommended) ----
CONCERT_LIB = $(CPLEX_PATH)/concert/lib/x86-64_linux/static_pic
CPLEX_LIB   = $(CPLEX_PATH)/cplex/lib/x86-64_linux/static_pic

LIBS = \
    -L$(GUROBI_HOME)/lib -lgurobi_c++ -lgurobi120 \
    $(CONCERT_LIB)/libconcert.a \
    $(CPLEX_LIB)/libilocplex.a \
    $(CPLEX_LIB)/libcplex.a \
    -lpthread -lm -ldl

# ---- Source Files (relative to this directory) ----
SRC = \
    IncrementalRMP.cpp \
    ColumnGenSolver.cpp \
    main.cpp

OBJ = $(SRC:.cpp=.o)

# ---- Output executable ----
TARGET = rlt1_cg

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