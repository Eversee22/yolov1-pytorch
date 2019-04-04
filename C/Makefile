DEBUG=0
OPENCV=1

EXEC=predict
OBJDIR=./obj/

CC=gcc
OPTS=-Ofast
LDFLAGS= -lm
COMMON=
CFLAGS=-Wall -Wfatal-errors

ifeq ($(DEBUG),1)
OPTS=-Og -g
endif

CFLAGS+=$(OPTS)

ifeq ($(OPENCV),1)
COMMON+= -DOPENCV #define Macro(-D)
CFLAG+= -DOPENCV
LDFLAGS+= `pkg-config --libs opencv`
COMMON+= `pkg-config --cflags opencv`
endif

OBJ=predict.o
OBJS = $(addprefix $(OBJDIR),$(OBJ))
DEPS = $(wildcard *.h) Makefile

all: obj $(EXEC)

$(EXEC):$(OBJS)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

obj:
	mkdir -p obj

.PHONY: clean

clean:
	rm -rf $(OBJS) $(EXEC)
