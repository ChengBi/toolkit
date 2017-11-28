#include "head.h"
#include "wordTree.h"
void printVector(vector<vector<string> > input){
    for (int i = 0; i < input.size(); i ++){
        for (int j = 0; j < input[i].size(); j ++){
            cout << input[i][j] << " ";
        }
        cout << endl;
    }
}

int main(){

    // dataProvider provider("../data/corpus3.txt");
    // printVector(provider.getWords());
    wordNode node1, node2, node3, node4, node5, node6;
    cout << node1.id << endl;
    cout << node2.id << endl;
    cout << node3.id << endl;
    cout << node4.id << endl;
    cout << node5.id << endl;
    cout << node6.id << endl;


    return 0;
}
