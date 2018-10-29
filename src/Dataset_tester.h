#ifndef TP2_METODOS_DATASET_TESTER_H
#define TP2_METODOS_DATASET_TESTER_H
#include <iostream>
#include <algorithm>    // std::sort
#include <vector>
#include <math.h> //fabs
#include "Dataset.h"

using namespace std;

class Dataset_tester {
    public:
        Dataset_tester(const Dataset& ds) : myDataset(ds){}
        void write_down_two_components();
        void checksumPCA();
    private:
    	Dataset myDataset;




};


#endif //TP2_METODOS_DATASET_TESTER_H