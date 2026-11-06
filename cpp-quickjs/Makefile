QUICKJS := quickjs
CSTD ?= -std=gnu11
CXXSTD ?= -std=c++20
CFLAGS ?= -O2 -g
CXXFLAGS ?= -O2 -g
QUICKJS_EXTRA_CFLAGS ?=

CPPFLAGS += -I$(QUICKJS)
COMMON_DEFINES := -D_GNU_SOURCE -DCONFIG_VERSION=\"$(shell cat $(QUICKJS)/VERSION.txt)\"
CFLAGS += $(COMMON_DEFINES) -fwrapv -funsigned-char -Wno-array-bounds -Wno-format-truncation $(QUICKJS_EXTRA_CFLAGS)
LDLIBS += -lm -lpthread

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
LDLIBS += -ldl
endif

BUILD := build
TARGET := $(BUILD)/mapgen-runner
QJS_OBJECTS := $(BUILD)/quickjs.o $(BUILD)/cutils.o $(BUILD)/libregexp.o $(BUILD)/libunicode.o $(BUILD)/dtoa.o
OBJECTS := $(QJS_OBJECTS) $(BUILD)/runner.o

.PHONY: all clean run run-test

all: $(TARGET)

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/%.o: $(QUICKJS)/%.c | $(BUILD)
	$(CC) $(CPPFLAGS) $(CSTD) $(CFLAGS) -c $< -o $@

$(BUILD)/runner.o: runner.cpp | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXSTD) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

DATA ?= ../../binaries/data/mods/public
SCRIPT ?= $(DATA)/maps/random/new_rms_test.js

run: $(TARGET)
	./$(TARGET) --data $(DATA) $(SCRIPT)

run-test: $(TARGET)
	./$(TARGET) --data $(DATA) $(DATA)/maps/random/tests/test_Generator.js

clean:
	rm -rf $(BUILD)
