#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>
#include <eigen>
#include <set>
#include <fstream>
#include <sstream>




using namespace std;

class Bigram{

public:
	Eigen::MatrixXf matrix;
	void loadCorpus(string filename);
};

Bigram::Bigram(){
	
}

void Bigram::loadCorpus(string filename){
	fstream file;
	file.open("../data/corpus.txt");
	string line;
	while(getline(file, line)){
		stringstream ss(line);
		string word;
		while(ss >> word){
			cout << word << endl;
		}
		ss.clear();
	}
	file.clear();
	file.close();
}




int main(){

	



	return 0;
}
