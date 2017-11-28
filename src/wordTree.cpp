#include "wordTree.h"
int IDgenerator(){
    return id ++;
}

dataProvider::dataProvider(string filename){
    ifstream file;
    file.open(filename.c_str());
    assert(file.is_open());
    string line;
    while(getline(file, line)){
     	stringstream ss(line);
    	string word;
        vector<string> temp;
    	while(ss >> word){
            temp.push_back(word);
        }
        ss.clear();
        this->words.push_back(temp);
    }
    file.clear();
    file.close();
}
vector<vector<string> > dataProvider::getWords(){
    return this->words;
}

wordNode::wordNode(){
    this->id = IDgenerator();
}
