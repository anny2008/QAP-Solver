# ============================================
# Makefile for rlt1_gurobi_reduction_IV (local to this folder)
# Directory: cpp/modules/rlt1_gurobi_reduction_IV
# Builds:   rlt1_gurobi_reduction_IV
# ============================================

# ---- Compiler ----
CXX      = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -fPIC
GUROBI_HOME = /opt/gurobi/gurobi1200/linux64

# ---- Include Paths ----
# We are in cpp/modules/rlt1_gurobi_reduction_IV, so go up two levels to reach core/include
INCLUDES = \
    -I$(GUROBI_HOME)/include \
    -I../../core \
    -I../../include \
    -I.


LIBS = \
    -L$(GUROBI_HOME)/lib -lgurobi_c++ -lgurobi120 \
    -lpthread -lm -ldl

#     -L$(GUROBI_HOME)/lib -lgurobi_c++ -lgurobi120 \
# ---- Source Files (relative to this directory) ----
SRC = \
    RLT1GurobiReductionIV.cpp \
    main.cpp

OBJ = $(SRC:.cpp=.o)

# ---- Output executable ----
TARGET = rlt1_gurobi_reduction_IV_solver

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