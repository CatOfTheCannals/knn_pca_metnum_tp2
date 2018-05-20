#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "gtest/gtest.h"
#include "../src/Dataset.h"

// --------- SET UP --------------
class datasetTest : public ::testing::Test {
protected:
    virtual void SetUp() {

    }
    Dataset holis = Dataset();

};

TEST_F (datasetTest, initializeDataset){
    holis.getImages();
    // std::cout << holis.getImages() << std::endl;

}
