cc=g++
exe=main
obj=main.o wordNode.o wordGraph.o
includedir=F:\github\toolkit\include
matdir=F:\github\eigen

$(exe):$(obj)
	$(cc) -o $(exe) $(obj)

wordNode.o:wordNode.cpp
	$(cc) -c wordNode.cpp -I $(includedir)

wordGraph.o:wordGraph.cpp
	$(cc) -c wordGraph.cpp -I $(includedir)

# main.o:main.cpp
#	$(cc) -c main.cpp -I $(includedir)

main.o:bigram.cpp
	$(cc) -o main.o -c bigram.cpp -I $(matdir) -I $(includedir)

clean:
	rm -rf $(obj) main
