#pragma once
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>
#include "wordNode.h"

class wordGraph{

    private:
		vector<int> ids;
		wordNode* head;   
 
	public:
		wordGraph();
		void insert(wordNode* node);
		void remove(wordNode* node);
		void combine(wordNode* node1, wordNode* node2);
		void print();
		void search();
};



