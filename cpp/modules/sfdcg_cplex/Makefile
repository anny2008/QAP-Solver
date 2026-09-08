# ============================================
# Makefile for sfdcg_cplex (local to this folder)
# Directory: cpp/modules/sfdcg_cplex
# Builds:   sfdcg_cplex
# ============================================

# ---- Compiler ----
CXX      = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -fPIC -fopenmp
CPLEX_PATH = /opt/ibm/ILOG/CPLEX_Studio2211
# ---- Include Paths ----
# We are in cpp/modules/sfdcg_cplex, so go up two levels to reach core/include
INCLUDES = \
    -I$(CPLEX_PATH)/concert/include \
    -I$(CPLEX_PATH)/cplex/include \
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
    -lpthread -lm -ldl

# ---- Source Files (relative to this directory) ----
SRC = \
    IncrementalRMP.cpp \
    PricingEngine.cpp \
    Solver.cpp \
    SubgraphDecomposition.cpp \
    main.cpp

OBJ = $(SRC:.cpp=.o)

# ---- Output executable ----
TARGET = sfdcg_cplex

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