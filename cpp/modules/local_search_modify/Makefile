CXX = g++
CXXFLAGS = -O2 -std=c++17
INCLUDES = -I../../core -I../../core/include
TARGET = local_search_solver
SRCS = main.cpp

all: $(TARGET)

$(TARGET): $(SRCS) local_search.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TARGET) $(SRCS)

clean:
	rm -f $(TARGET)
