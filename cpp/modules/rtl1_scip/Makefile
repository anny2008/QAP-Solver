# Makefile for RTL1 SCIP Solver

CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -fopenmp

# Try local SCIP build first, fall back to system SCIP
SCIP_PATH ?= /home/local.isima.fr/antran/UFF/scip-9.2.0
SCIP_LIBDIR = $(SCIP_PATH)/lib

# Check which SCIP library is available
SCIP_LIB = $(SCIP_LIBDIR)/shared/libscip.so
ifeq (,$(wildcard $(SCIP_LIB)))
    # Try static library
    SCIP_LIB = $(SCIP_LIBDIR)/static/libscip.a
    ifeq (,$(wildcard $(SCIP_LIB)))
        # Fall back to system SCIP
        SCIP_PATH = /home/local.isima.fr/antran/UFF/SCIPOptSuite-9.2.0-Linux
        SCIP_LIB = $(SCIP_PATH)/lib/libscip.so
    endif
endif

SCIP_INCDIR = $(SCIP_PATH)/lib/include

# Include and library flags
INCFLAGS = -I$(SCIP_INCDIR)
LIBFLAGS = -L$(dir $(SCIP_LIB)) -lscip -lpthread -ldl -lm

# Source and output
SRC = rtl1_solver.cpp
OBJ = rtl1_solver.o
EXE = rtl1_solver

# Build rules
all: $(EXE)

$(EXE): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIBFLAGS)

$(OBJ): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(EXE)

.PHONY: all clean
