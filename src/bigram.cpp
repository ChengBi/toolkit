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
#include <stack>



using namespace std;

class Bigram{

public:
	Bigram(string filename);
	void graphGenerator();
	void directGraphGenerator();
	void printSet();
	void printMap();
	void getPath();
	~Bigram();
	void search(wordNode* head, wordNode* tail);

private:
	set<string> words_set;
	map<int, string> words_map;
	map<string, int> words_map_inverse;
	map<pair<int, int>, int> graph;
	vector<wordNode*> starters;
	vector<wordNode*> enders;
	vector<wordNode*> directedGraph;
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
	// map<string, int>::iterator iti;
	// for (iti = this->words_map_inverse.begin(); iti != this->words_map_inverse.end(); ++iti){
	// 	cout << iti->first << " => " << iti->second <<endl;
	// }
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
		printf("Key: %d => %d , Value: %d \n",it->first.first, it->first.second, it->second);
	}
}

void Bigram::directGraphGenerator(){
	// vector<wordNode*> nodes;
	map<int, string>::iterator it;
	for (it = this->words_map.begin(); it != this->words_map.end(); ++it){
		this->directedGraph.push_back(new wordNode(it->second, it->first));
		// nodes[nodes.size()-1]->toString();
	}
	printf("Building directed graph ...\n");
	for (int i = 0; i < this->directedGraph.size(); i ++){
		for (int j = 0; j < this->directedGraph.size(); j ++){
			if (i != j ){
				pair<int, int> key(this->directedGraph[i]->getID(), this->directedGraph[j]->getID());
				if (this->graph.find(key) != this->graph.end()){
					this->directedGraph[i]->addSon(this->directedGraph[j]);
					this->directedGraph[j]->addParent(this->directedGraph[i]);
				}
			}
		}
	}
	printf("Built target ... \n");
	for(int i = 0; i < this->directedGraph.size(); i ++){
		this->directedGraph[i]->checkSelf();
		if (this->directedGraph[i]->getType() == 0)
			this->starters.push_back(this->directedGraph[i]);
		if (this->directedGraph[i]->getType() == 2)
			this->enders.push_back(this->directedGraph[i]);
	}
	// for(int i = 0; i < this->directedGraph.size(); i ++){
	// 	this->directedGraph[i]->toString();
	// }
}

void Bigram::getPath(){
	for (int i = 0; i < this->starters.size(); i ++){
		for (int j = 0; j < this->enders.size(); j ++){
			printf("Searching nodes: %d => %d \n", this->starters[i]->getID(), this->enders[j]->getID());
			search(this->starters[i], this->enders[j]);
		}
	}
	// search(this->starters[0], this->enders[0]);
}
/*
	search function cannot handle the following situation:
	e.g. a e, where 'a' and 'e' are head node and tail node respectively.
	e.g. a, where a senence has only only word.
*/
void Bigram::search(wordNode* head, wordNode* tail){
	stack<wordNode*> s;
	vector<vector<int> > paths;
	vector<int> path_temp;
	printf("Searching Progress ...\n");
	s.push(head);
	vector<int> previous;
	while(!s.empty()){
		if (s.size() == 1){
			previous.push_back(s.top()->getID());
		}
		// for (int i = 0; i < previous.size(); i ++){
		// 	path_temp.push_back(previous[i]);
		// }
		// printf("ID: %d \n", s.top()->getID());
		// for (int i = 0; i < path_temp.size(); i ++){
			// cout << path_temp[i] << " ";
		// }
		// cout << s.size() << endl;
		wordNode* top = s.top();
		vector<wordNode*> children = top->sons;
		s.pop();
		path_temp.push_back(top->getID());
		if (top->getType() == 2){
			if (top->getID() == tail->getID()){
				paths.push_back(path_temp);
			}
			// for (int i = 0; i < path_temp.size(); i ++){
			// 	cout << path_temp[i] << " ";
			// }
			// cout << " => " ;
			// for (int i = 0; i < previous.size(); i ++){
			// 	cout << previous[i] << " ";
			// }
			// cout << endl;
			path_temp.clear();
		}
		// else{
		set<int> ids;
		for (int i = 0; i < path_temp.size(); i ++){
			ids.insert(path_temp[i]);
		}
		if (ids.size() != path_temp.size()){
			path_temp.clear();
				// s.pop();
			// continue;
		}
		// else{
			for (int i = 0; i < children.size(); i ++){

				s.push(children[i]);
			}
		// }
	}
	for (int i = 0; i < paths.size(); i ++){
		for (int j = 0; j < paths[i].size(); j ++){
			cout << paths[i][j] << " ";
		}
		cout << endl;
	}
}

Bigram::~Bigram(){
	words_set.clear();
	words_map.clear();
	words_map_inverse;
	graph.clear();
}

// int main(){
//
// 	Bigram bigram("../data/corpus3.txt");
// 	bigram.printMap();
// 	bigram.directGraphGenerator();
// 	bigram.getPath();
// 	// map<int, int> temp;
// 	// temp.insert(pair<int, int>(1,3));
// 	// temp.insert(pair<int, int>(2,5));
// 	// map<int ,int >::iterator it;
// 	// for (it = temp.begin(); it != temp.end(); ++it)
// 	// 	cout << it->first << " => " << it->second<< endl;
// 	// cout << temp.find(1)->second << endl;
//
//
//
//
// 	return 0;
// }
