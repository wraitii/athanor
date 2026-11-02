SPIDERMONKEY ?= ../../libraries/source/spidermonkey
CXX ?= clang++
CXXFLAGS ?= -std=c++20 -O2 -g
CPPFLAGS += -isystem$(SPIDERMONKEY)/include-release
LDFLAGS += -L$(SPIDERMONKEY)/lib
LDLIBS += -lmozjs128-release -lmozjs128-rust -lz -lpthread

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
LDLIBS += -framework Security -framework CoreFoundation
else
LDLIBS += -ldl -lm
endif

TARGET := build/mapgen-runner
SOURCE := runner.cpp

.PHONY: all clean run run-test

all: $(TARGET)

$(TARGET): $(SOURCE)
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

DATA ?= ../../binaries/data/mods/public
SCRIPT ?= $(DATA)/maps/random/new_rms_test.js

run: $(TARGET)
	./$(TARGET) --data $(DATA) $(SCRIPT)

run-test: $(TARGET)
	./$(TARGET) --data $(DATA) $(DATA)/maps/random/tests/test_Generator.js

clean:
	rm -rf build
