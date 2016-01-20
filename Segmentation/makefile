INCDIR = -I.
DBG    = -g
OPT    = -O3
CPP    = g++
CFLAGS = $(OPT) $(INCDIR)
OPENCV = `pkg-config --libs opencv`
LINK   = -lm $(OPENCV)

.cpp.o:
	$(CPP) $(CFLAGS) -c $< -o $@

all: seg_meanshift seg_graph

seg_meanshift: seg_meanshift.cpp
	$(CPP) $(CFLAGS) -o seg_meanshift seg_meanshift.cpp $(LINK)

seg_graph: seg_graph.cpp segment-image.h segment-graph.h disjoint-set.h
	$(CPP) $(CFLAGS) -o seg_graph seg_graph.cpp $(LINK)

clean:
	rm -f *~
	rm -f seg_graph
	rm -f seg_meanshift

clean-all: clean
	rm -f *~ 
	rm -f seg_graph
	rm -f seg_meanshift