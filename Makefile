CC = gcc
CFLAGS = -Wall -Wextra -O2 -fopenmp -Iinc
LDFLAGS = -lm
SRC_DIR = src
INC_DIR = inc
BUILD_DIR = build
BIN_DIR = bin
SRC_FILES = $(wildcard $(SRC_DIR)/*.c)
OBJ_FILES = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(SRC_FILES))
TARGET = $(BIN_DIR)/neural_net

all: $(TARGET)

$(TARGET): $(OBJ_FILES) $(BUILD_DIR)/main.o | $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/main.o: main.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR) *.o $(TARGET)

.PHONY: all clean
