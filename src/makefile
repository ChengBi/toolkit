cc=g++
exe=main
obj=main.o wordNode.o wordGraph.o
includedir=F:\\github\\toolkit\\include\\

$(exe):$(obj)
	$(cc) -o $(exe) $(obj)

wordNode.o:wordNode.cpp
	$(cc) -c wordNode.cpp -I $(includedir)
	$(cc) -c wordGraph.cpp -I $(includedir)

main.o:main.cpp
	$(cc) -c main.cpp -I $(includedir)

clean:
	rm -rf $(obj) main



