# Makefile for SFD SCIP solver (alternative to CMake)

CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall
SCIPDIR = /home/local.isima.fr/antran/UFF/scip-9.2.0

# SCIP include and library paths
INCLUDES = -I$(SCIPDIR)/src -I../../core
LDFLAGS = -L$(SCIPDIR)/lib
LIBS = -lscip

TARGET = sfd_solver
SRC = sfd_solver.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) $(LIBS)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean
