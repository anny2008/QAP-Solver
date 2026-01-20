CXX = g++
CXXFLAGS = -O2 -std=c++17 -I../../core -I../../core/include

all: ga_solver

ga_solver: main.cpp ga_solver.cpp ga_solver.hpp ../../core/problem.h ../../core/solution.h
	$(CXX) $(CXXFLAGS) main.cpp ga_solver.cpp -o ga_solver

clean:
	rm -f ga_solver
