CC=g++
#CC=g++-4.8
#CC=g++-5 #significantly faster code compared to g++-4
#CC=g++-7

#DISABLEWARN = -Wno-unused-but-set-variable -Wno-unused-result
#ARCHFLAGS =  -march=x86-64  -mno-avx
#CXXFLAGS +=  -O3 $(ARCHFLAGS)  -Wall -static -I.  -std=c++11

#CXXFLAGS +=  -O3 $(ARCHFLAGS)  -Wall -I.  -std=c++11
CXXFLAGS +=  -O3 $(ARCHFLAGS)  -Wall -I.  -std=c++11 -g -static
LIBS = -lpthread

CFLAGS = -O3

DEPS = *.cpp *.h
OBJS=argtable3.o options.o
.PHONY:	all clean

PROGS= rknng

all: rknng

#Argtables should support compiling with g++, but there was a error message.
argtable3.o:
	gcc -c $(CFLAGS) contrib/argtable3.c

options.o:
	$(CC) -c $(CXXFLAGS) options.c

rknng: $(DEPS) $(OBJS)
	$(CC) $(CXXFLAGS) $(DISABLEWARN) knng.cpp $(LIBS) $(OBJS) -o rknng 

clean:
	rm -f $(PROGS) *.o

#	g++ -O3 -c -std=c++11 -fPIC -o rknng_lib.o rknng_lib.cpp
apitest:
	$(CC) -O3 -c -std=c++11 -o rknng_lib.o rknng_lib.cpp $(LIBS)
	gcc -c options.c
	gcc -c apitest.c
	$(CC) -o apitest apitest.o rknng_lib.o options.o $(LIBS)

