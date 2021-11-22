# Makefile for the SPAM system
# Written by John Wallin
# Jun 9, 1997

# may have to choose the compiler and
# compilation flags appropriate for your system


# Harlie settings
F77 = g77
F90 = gfortran -g 
LIBS = -L/usr/local/pgplot_gnu/pgplot -lpgplot -lpng -lz -L/usr/X11R6/lib -lX11
FLAG = 

# Macintosh settings
F77 = gfortran
F90 = gfortran -g 
LIBS = -L/sw/lib/pgplot -lpgplot -L/sw/lib -lpng -lz \
	-L/usr/X11R6/lib -lX11 -L/sw/lib -laquaterm \
	-Wl,-framework -Wl,Foundation
FLAG = 


# skynet settings
F77 = gfortran
F90 = gfortran -pg -fbounds-check -Wall -fimplicit-none
F90 = gfortran -O3
LIBS = -L/usr/local/pgplot_gnu/pgplot -lpgplot -lpng -lz -L/usr/X11R6/lib -lX11
FLAG = 


RM= /bin/rm -f
#--- You shouldn't have to edit anything else. ---


all: standard 
standard: basic_run 
unp: basic_run_unpreturbed
unp2: unpreturbed_only
c: test.o

# process_pnm
####reproduce
#########

##########

.c.o: 
	$(CC) $(CFLAGS) -c $<

.cc.o:
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f basic_run *.o *.mod 

veryclean:
	rm -f basic_run idriver *.o *.mod process_pnm process regenerate fstates trajectory zoo_reproduce gmon.out a.* plt *.jpg *.mpg fort.* pfile mpeg_parameters

process_pnm: process_pnm.c
	$(CC) -o process_pnm process_pnm.c

init_module.o: init_module.f90
	$(F90) -c -g init_module.f90

parameters_module.o: parameters_module.f90
	$(F90) -c -g parameters_module.f90

fitness_module.o: fitness_module.f90
	$(F90) -c -g fitness_module.f90

setup_module.o:	setup_module.f90 parameters_module.o
	$(F90) -c -g setup_module.f90

io_module.o: io_module.f90 parameters_module.o
	$(F90) -c -g io_module.f90

integrator.o:  integrator.f90 parameters_module.o
	$(F90) -c -g integrator.f90

align_module.o:  align_module.f90
	$(F90) -c -g align_module.f90

df_module.o:  df_module.f90
	$(F90) -c -g df_module.f90

basic_run.o: basic_run.f90 parameters_module.o
	$(F90) -c -g basic_run.f90

basic_run_unpreturbed.o: basic_run_unpreturbed.f90 parameters_module.o
	$(F90) -c -g basic_run_unpreturbed.f90

unpreturbed_only.o: unpreturbed_only.f90 parameters_module.o
	$(F90) -c -g unpreturbed_only.f90



################


basic_run: parameters_module.o io_module.o \
	df_module.o setup_module.o \
	integrator.o init_module.o basic_run.o 
	$(F90) -o basic_run -g basic_run.o parameters_module.o \
	df_module.o setup_module.o io_module.o  \
	 init_module.o integrator.o 

basic_run_unpreturbed: parameters_module.o io_module.o \
	df_module.o setup_module.o \
	integrator.o init_module.o basic_run_unpreturbed.o 
	$(F90) -o basic_run_unpreturbed -g basic_run_unpreturbed.o parameters_module.o \
	df_module.o setup_module.o io_module.o  \
	 init_module.o integrator.o 

unpreturbed_only: parameters_module.o io_module.o \
	df_module.o setup_module.o \
	integrator.o init_module.o unpreturbed_only.o 
	$(F90) -o unpreturbed_only -g unpreturbed_only.o parameters_module.o \
	df_module.o setup_module.o io_module.o  \
	 init_module.o integrator.o 


#align_module.o

sPlot: a.101
	gnuplot GalaxyScatterPlot.plt
	rm ~/public_html/GalSPlot.png
	cp GalSPlot.png ~/public_html/
	chmod 777 ~/public_html/GalSPlot.png

binCntPlot: TestData.f90
	gfortran TestData.f90
	./a.out > BinCnt.out
	gnuplot GalaxyBinCntPlot.plt
	rm ~/public_html/GalBinCntPlot.png
	cp GalBinCntPlot.png ~/public_html/
	chmod 777 ~/public_html/GalBinCntPlot.png

test.o: Metropolis_pThreads.cpp
	g++ -o test.o Metropolis_pThreads.cpp -lpthread

