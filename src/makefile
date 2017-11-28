cc=g++
exe=main
obj=main.o wordTree.o #wordNode.o wordGraph.o bigram.o
includedir=F:\github\toolkit\include
matdir=F:\github\eigen

$(exe):$(obj)
	$(cc) -o $(exe) $(obj)

wordNode.o:wordNode.cpp
	$(cc) -c wordNode.cpp -I $(includedir)

wordGraph.o:wordGraph.cpp
	$(cc) -c wordGraph.cpp -I $(includedir)

bigram.o:bigram.cpp
	$(cc) -c bigram.cpp -I $(matdir) -I $(includedir)

wordTree.o:wordTree.cpp
	$(cc) -c wordTree.cpp -I $(includedir)

main.o:main.cpp
	$(cc) -c main.cpp -I $(includedir)

clean:
	rm -rf $(obj) main
