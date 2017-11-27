#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <set>
#include <fstream>
#include <sstream>
#include <map>
#include "wordNode.h"



using namespace std;

class Bigram{

public:
	Bigram(string filename);
	void graphGenerator();
	void directGraphGenerator();
	void printSet();
	void printMap();
	~Bigram();

private:
	set<string> words_set;
	map<int, string> words_map;
	map<string, int> words_map_inverse;
	map<pair<int, int>, int> graph;
	int size;
	string filename;
	void loadCorpus();


};

Bigram::Bigram(string filename){
	this->filename = filename;
	loadCorpus();
	graphGenerator();
}

void Bigram::loadCorpus(){
	fstream file;
	file.open(this->filename.c_str());
	string line;
	while(getline(file, line)){
		stringstream ss(line);
		string word;
		while(ss >> word){
			// cout << word << endl;
			this->words_set.insert(word);
		}
		ss.clear();
	}
	file.clear();
	file.close();
	set<string>::iterator it;
	int index = 0;
	for (it = this->words_set.begin(); it != this->words_set.end(); ++it)
	{
		this->words_map.insert(pair<int, string>(index, *it));
		this->words_map_inverse.insert(pair<string, int>(*it, index));
		index ++;
	}
	size = this->words_set.size();
	// cout << size << endl;
}
void Bigram::printSet(){
	set<string>::iterator it;
	for (it = this->words_set.begin(); it != this->words_set.end(); ++it)
	{
		cout << *it << endl;
	}
}
void Bigram::printMap(){
	map<int, string>::iterator it;
	for (it = this->words_map.begin(); it != this->words_map.end(); ++it){
		cout << it->first << " => " << it->second <<endl;
	}
	map<string, int>::iterator iti;
	for (iti = this->words_map_inverse.begin(); iti != this->words_map_inverse.end(); ++iti){
		cout << iti->first << " => " << iti->second <<endl;
	}
}

void Bigram::graphGenerator(){
	fstream file;
	file.open(this->filename.c_str());
	string line;
	while(getline(file, line)){
		stringstream ss(line);
		string word;
		vector<string> words;
		while(ss >> word){
			// cout << word << endl;
			words.push_back(word);
		}
		for (int i = 0; i < words.size()-1; i ++){
			// this->graph.insert();
			pair<int, int> key(this->words_map_inverse.find(words[i])->second,
					this->words_map_inverse.find(words[i+1])->second);
			if (this->graph.find(key) == this->graph.end())
				this->graph.insert(pair<pair<int, int>, int>(key, 0));

			this->graph[pair<int, int>(this->words_map_inverse.find(words[i])->second,
							this->words_map_inverse.find(words[i+1])->second)] += 1;
		}
		ss.clear();
	}
	file.clear();
	file.close();
	map<pair<int, int>, int>::iterator it;
	for (it = this->graph.begin(); it != this->graph.end(); ++it){
		cout << "Key: " << it->first.first << " - " << it->first.second << " , Value: " << it->second << endl;
	}
}

void Bigram::directGraphGenerator(){
	vector<wordNode*> nodes;
	map<int, string>::iterator it;
	for (it = this->words_map.begin(); it != this->words_map.end(); ++it){
		nodes.push_back(new wordNode(it->second, it->first));
		// nodes[nodes.size()-1]->toString();
	}
	for (int i = 0; i < nodes.size(); i ++){
		for (int j = 0; j < nodes.size(); j ++){
			if (i != j ){
				pair<int, int> key(nodes[i]->getID(), nodes[j]->getID());
				if (this->graph.find(key) != this->graph.end()){
					nodes[i]->addSon(nodes[j]);
					nodes[j]->addParent(nodes[i]);
				}
			}
		}
	}
	for(int i = 0; i < nodes.size(); i ++){
		nodes[i]->checkSelf();
	}
	for(int i = 0; i < nodes.size(); i ++){
		nodes[i]->toString();
	}
}

Bigram::~Bigram(){
	words_set.clear();
	words_map.clear();
	words_map_inverse;
	graph.clear();
}

int main(){

	Bigram bigram("../data/corpus.txt");
	bigram.printMap();
	bigram.directGraphGenerator();
	// map<int, int> temp;
	// temp.insert(pair<int, int>(1,3));
	// temp.insert(pair<int, int>(2,5));
	// map<int ,int >::iterator it;
	// for (it = temp.begin(); it != temp.end(); ++it)
	// 	cout << it->first << " => " << it->second<< endl;
	// cout << temp.find(1)->second << endl;




	return 0;
}
