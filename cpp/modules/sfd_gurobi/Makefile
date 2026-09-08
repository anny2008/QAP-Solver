# ============================================
# Makefile for sfd_gurobi (local to this folder)
# Directory: cpp/modules/sfd_gurobi
# Builds:   sfd_gurobi
# ============================================

# ---- Compiler ----
CXX      = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -fPIC
GUROBI_HOME = /opt/gurobi/gurobi1200/linux64

# ---- Include Paths ----
# We are in cpp/modules/sfd_gurobi, so go up two levels to reach core/include
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
    SFDGurobi.cpp \
    main.cpp

OBJ = $(SRC:.cpp=.o)

# ---- Output executable ----
TARGET = sfd_gurobi_solver

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