#pragma once
#include "head.h"
static int id = 0;
class dataProvider{

public:
    dataProvider(string filename);
    vector<vector<string> > getWords();
private:
    vector<vector<string> > words;
};

class wordNode{

public:
    int id;
    wordNode();

};

class wordTree{

};
