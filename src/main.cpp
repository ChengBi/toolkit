#include "wordNode.h"
#include <fstream>
#include <cassert>
#include <sstream>


int main(){

    ifstream file;
    file.open("../data/line.txt");
    assert(file.is_open());
    string line;
    while(getline(file, line)){
    //	file >> word;
     	stringstream ss(line);
	string word;
	while(ss >> word){
	    cout << "add node: " << word << endl;
	    wordNode* node = new wordNode(word);
	}
        ss.clear();    	
//	printf(word.c_str());
//	printf("\n");
    }

    file.clear();
    file.close();
    return 0;
}
