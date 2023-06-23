# Compiler
FC = gfortran

# Compiler flags
FCFLAGS = -g -Wall -O2

# Target executable
TARGET = parameter_sweep

# Object files
OBJS = PhotosynthesisMod.o parameter_sweep.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(FC) $(FCFLAGS) -o $(TARGET) $(OBJS)

PhotosynthesisMod.o: PhotosynthesisMod.f90
	$(FC) $(FCFLAGS) -c PhotosynthesisMod.f90

parameter_sweep.o: parameter_sweep.f90 PhotosynthesisMod.o
	$(FC) $(FCFLAGS) -c parameter_sweep.f90

.PHONY: clean

clean:
	rm -f *.o *.mod $(TARGET)
