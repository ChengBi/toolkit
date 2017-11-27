#include "wordNode.h"

wordNode::wordNode(string word, int id){
    this->word = word;
    this->id = id;
    //this->parents = vector<wordNode*>();
    //this->sons = vector<wordNode*>();
    checkSelf();
}

wordNode::wordNode(string word){
    wordNode(word, -1);
}

wordNode::wordNode(){
    wordNode("");
}

void wordNode::addParent(wordNode* parent){
    this->parents.push_back(parent);
//    infinite loop
//    parent->addSon(this);
    // checkSelf();
}

void wordNode::delParent(wordNode* parent){

}

void wordNode::addSon(wordNode* son){
    this->sons.push_back(son);
//    infinite loop
//    son->addParent(this);
    // checkSelf();
}

void wordNode::delSon(wordNode* son){

}


void wordNode::checkSelf(){
    if (this->parents.size() == 0 and this->sons.size() == 0){
        this->type = BAD_NODE;
    }
    else if (this->parents.size() == 0 and this->sons.size() != 0){
        this->type = START;
    }
    else if (this->parents.size() != 0 and this->sons.size() == 0){
        this->type = END;
    }
    else{
        this->type = CONNECTOR;
    }
}

NODE_TYPE wordNode::getType(){
    return this->type;
}

int wordNode::getNeighboursCount(){
    return this->neighbours_count;
}

string wordNode::getWord(){
    return this->word;
}

int wordNode::getID(){
    return this->id;
}

void wordNode::toString(){
    printf("|-------------------------------|\n");
    printf("|           NODE INFO           |\n");
    printf("|-------------------------------|\n");
    printf("| word:          %s              \n", this->word.c_str());
    printf("| parents count: %d              \n", this->parents.size());
    printf("| sons count:    %d              \n", this->sons.size());
    printf("| id:            %d              \n", this->id);
    printf("| type:          %d              \n", this->type);
    printf("|_______________________________|\n");
}

wordNode::~wordNode(){
    this->parents.clear();
    this->sons.clear();
}
