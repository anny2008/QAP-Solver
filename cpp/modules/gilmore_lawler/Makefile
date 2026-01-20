CXX = g++
CXXFLAGS = -O2 -std=c++17
INCLUDES = -I../../core -I../../core/include
TARGET = gilmore_lawler_solver
SRCS = main.cpp gilmore_lawler.cpp

all: $(TARGET)

$(TARGET): $(SRCS) gilmore_lawler.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TARGET) $(SRCS)

clean:
	rm -f $(TARGET)
