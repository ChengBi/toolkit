#pragma once
#include <stdio.h>
#include <vector>
#include <iostream>
#include <string>
using namespace std;

enum NODE_TYPE{
    START, CONNECTOR, END, BAD_NODE
};

class wordNode{

    private:

		string word;
        int neighbours_count;
        int id;
        NODE_TYPE type;
        // vector<wordNode*> parents;
        // vector<wordNode*> sons;
		// void checkSelf();
		vector<int> optional_ids;

    public:
        vector<wordNode*> parents;
	    vector<wordNode*> sons;

        wordNode(string word, int id);
        wordNode(string word);
        wordNode();

		void addParent(wordNode* parent);
		void delParent(wordNode* parent);
		void addSon(wordNode* son);
		void delSon(wordNode* son);
        int getID();
	    void checkSelf();
		NODE_TYPE getType();
		int getNeighboursCount();
		string getWord();
		void toString();
		~wordNode();

};
