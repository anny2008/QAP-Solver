CXX = g++
CXXFLAGS = -O2 -std=c++17 -I../../core -I../../core/include

all: vns_solver

vns_solver: main.cpp vns_solver.cpp vns_solver.hpp ../../core/problem.h ../../core/solution.h
	$(CXX) $(CXXFLAGS) main.cpp vns_solver.cpp -o vns_solver

clean:
	rm -f vns_solver
