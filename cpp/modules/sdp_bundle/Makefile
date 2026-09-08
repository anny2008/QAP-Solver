
.DEFAULT_GOAL := all

CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -fopenmp -fPIC


CPLEX_PATH = /opt/ibm/ILOG/CPLEX_Studio2211
GUROBI_HOME = /opt/gurobi/gurobi1200/linux64
# ---- Include Paths ----
# We are in cpp/modules/cg_cplex, so go up two levels to reach core/include
BUNDLE_INC_PATH = /home/local.isima.fr/antran/local/include/NDOFiOracle
OSI_INC_PATH = /home/local.isima.fr/antran/local/include/coin_or
EIGEN_INC_PATH = /home/local.isima.fr/antran/.local/include
MOSEK_PATH = /home/local.isima.fr/antran/mosek/mosek/11.2/tools/platform/linux64x86
SCS_PATH = /home/local.isima.fr/antran/local/scs
INCLUDES = \
    -I$(BUNDLE_INC_PATH) \
    -I$(OSI_INC_PATH) \
    -I$(EIGEN_INC_PATH) \
    -I$(MOSEK_PATH)/h \
    -I$(CPLEX_PATH)/concert/include \
    -I$(CPLEX_PATH)/cplex/include \
    -I../../core \
    -I../../include \
    -I. 

# ---- Library Paths (static PIC recommended) ----
CONCERT_LIB = $(CPLEX_PATH)/concert/lib/x86-64_linux/static_pic
CPLEX_LIB   = $(CPLEX_PATH)/cplex/lib/x86-64_linux/static_pic
BUNDLE_LIB = /home/local.isima.fr/antran/local/lib/libNDOFiOracle.a
SCS_LIB = /home/local.isima.fr/antran/local/lib/libscs.a
# OSI_LIB = /home/local.isima.fr/antran/local/lib/libOsiCpx.a

LIBS = \
    $(BUNDLE_LIB) \
    -L$(MOSEK_PATH)/bin \
    -lfusion64 -lmosek64 \
    $(CONCERT_LIB)/libconcert.a \
    $(CPLEX_LIB)/libilocplex.a \
    $(CPLEX_LIB)/libcplex.a \
    -lpthread -lm -ldl

SRC = \
    Hungarian.cpp \
    main.cpp \
    sdp_bundle_solver.cpp

OBJ = $(SRC:.cpp=.o)

# ---- Output executable ----
TARGET = sdp_bundle_solver

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