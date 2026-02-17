CXX      := g++
CXXFLAGS := -std=c++17 -O2 -fPIC -Wall -Wextra
LDFLAGS  := -shared -L$(CURDIR)/deps/lib
LIBS     := -larmadillo -llapack -lopenblas

# Header-only deps: add include paths
MLPACK_DIR := $(wildcard $(CURDIR)/deps/mlpack-*/src)
ARMA_DIR   := $(wildcard $(CURDIR)/deps/armadillo-*/include)
ENS_DIR    := $(wildcard $(CURDIR)/deps/ensmallen-*/include)
CEREAL_DIR := $(wildcard $(CURDIR)/deps/cereal-*/include)

ifneq ($(MLPACK_DIR),)
  CXXFLAGS += -I$(MLPACK_DIR)
endif
ifneq ($(ARMA_DIR),)
  CXXFLAGS += -I$(ARMA_DIR)
endif
ifneq ($(ENS_DIR),)
  CXXFLAGS += -I$(ENS_DIR)
endif
ifneq ($(CEREAL_DIR),)
  CXXFLAGS += -I$(CEREAL_DIR)
endif

# Suppress warnings from header-only libs
CXXFLAGS += -Wno-unused-parameter -Wno-deprecated-declarations

TARGET   := libjmlpack.so
SRC      := jmlpack.cpp
OBJ      := jmlpack.o

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

$(OBJ): $(SRC) jmlpack.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(TARGET)
