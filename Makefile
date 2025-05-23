CC = gcc
CFLAGS = -O3 -Wall -Werror -pthread
LIBS= -lgsl -lm

all: Gaussian_poolOrNot BetaBinomial_Jeffreys_sample Gaussian_poolOrNot_cli

OBJS := GSLfun.o

deps := $(OBJS:%.o=.%.o.d)

# Control the build verbosity
ifeq ("$(VERBOSE)","1")
    Q :=
    VECHO = @true
else
    Q := @
    VECHO = @printf
endif

%.o: %.c %.h
	$(VECHO) "  CC\t$@\n"
	$(Q)$(CC) -o $@ $(CFLAGS) -c -MMD -MF .$@.d $< $(LIBS)


Gaussian_poolOrNot: $(OBJS) Gaussian_poolOrNot.c thread_setup.h
	$(VECHO) "  LD\t$@\n"
	$(Q)$(CC) $(CFLAGS) -o $@ $^ $(LIBS)


Gaussian_poolOrNot_cli: $(OBJS) Gaussian_poolOrNot_cli.c thread_setup.h
	$(VECHO) "  LD\t$@\n"
	$(Q)$(CC) $(CFLAGS) -o $@ $^ $(LIBS)


BetaBinomial_Jeffreys_sample: $(OBJS) BetaBinomial_Jeffreys_sample.c thread_setup.h
	$(VECHO) "  LD\t$@\n"
	$(Q)$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -f $(OBJS) $(deps)
	rm -f Gaussian_poolOrNot Gaussian_poolOrNot_cli
	rm -f BetaBinomial_Jeffreys_sample

.PHONY: all clean
