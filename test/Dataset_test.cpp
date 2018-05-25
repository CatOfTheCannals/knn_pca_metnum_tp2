#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "chrono"
#include "gtest/gtest.h"
#include "../src/Dataset.h"
#include "../src/Pca.h"

#define GET_TIME std::chrono::high_resolution_clock::now()
#define GET_TIME_DELTA(begin, end) \
     std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()

// --------- SET UP --------------
class datasetTest : public ::testing::Test {
protected:
    virtual void SetUp() {

        g.setIndex(0, 0, 1);
        g.setIndex(0, 1, 2);
        g.setIndex(0, 2, 3);

        g.setIndex(1, 0, 6);
        g.setIndex(1, 1, 4);
        g.setIndex(1, 2, 5);

        g.setIndex(2, 0, 9);
        g.setIndex(2, 1, 7);
        g.setIndex(2, 2, 8);

        l.setIndex(1, 0, 2);
        l.setIndex(2, 0, 1);
        l.setIndex(0, 0, 1);

        littleMock = Dataset(g, l);
    }
    Dataset reduced = Dataset("../../test/casos_test/", "testRed.in");

    Matrix l = Matrix(3,1);
    Matrix g = Matrix(3,3);
    Dataset littleMock;
};

/*
TEST_F (datasetTest, shuffle) {
    int n = 20;

    std::vector<int> numbers;
    for(int i = 0; i < n; i++) {
        numbers.push_back(i);
    }

    srand (time(NULL));
    for(int i = 0; i < n; i++) {

        int swap_index = rand() % n - i ;
        iter_swap(numbers.begin() + i, numbers.begin() + i + swap_index);
    }
    for(int i = 0; i < n; i++) {
        std::cout << numbers[i] << ", ";

    }
    std::cout << std::endl;
}
*/
/*
TEST_F (datasetTest, littleKnn) {
    littleMock.splitTrainFromTest(0.3);
    std::cout << littleMock.kNN_predict(1) << std::endl;
}

TEST_F (datasetTest, littlePca) {
    littleMock.trainPca(3, 0.0001);
    std::cout << littleMock.getPcaVecs() << std::endl;
    std::cout << littleMock.getPcaLambdas() << std::endl;

}
*/
TEST_F (datasetTest, reducedKnn) {
    reduced.shuffle();
    reduced.splitTrainFromTest(0.3);
    reduced.kNN_predict(10);
}

/*
TEST_F (datasetTest, reducedPca) {
    reduced.trainPca(3, 0.0001);
    std::cout << reduced.getPcaVecs() << std::endl;
    std::cout << reduced.getPcaLambdas() << std::endl;

}*/