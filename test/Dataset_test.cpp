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

        e.setIndex(0, 0, 1);
        e.setIndex(0, 1, 2);
        e.setIndex(0, 2, 3);
        e.setIndex(1, 0, 4);
        e.setIndex(1, 1, 5);
        e.setIndex(1, 2, 6);
        e.setIndex(2, 0, 7);
        e.setIndex(2, 1, 8);
        e.setIndex(2, 2, 9);
        e.setIndex(3, 0, 10);
        e.setIndex(3, 1, 11);
        e.setIndex(3, 2, 12);
        e.setIndex(4, 0, 13);
        e.setIndex(4, 1, 14);
        e.setIndex(4, 2, 15);

        l.setIndex(1, 0, 2);
        l.setIndex(2, 0, 1);
        l.setIndex(0, 0, 1);

        littleMock = Dataset(g, l);
    }

    Dataset reduced = Dataset("../../test/casos_test/", "testRed.in");
    Dataset big = Dataset("../../test/casos_test/", "testBig.in");

    Matrix l = Matrix(3,1);
    Matrix g = Matrix(3,3);
    Matrix e = Matrix(5,3);
    Dataset littleMock;
};
/*
TEST_F (datasetTest, lab) {

    int first_row, last_row;
    int k = 3;
    reduced.shuffle();
    reduced.kFold(first_row, last_row, k);
}
*/
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
/*
TEST_F (datasetTest, reducedKnn) {
    reduced.shuffle();
    //reduced.splitTrainFromTest(0.3);
    auto predicted_labels = reduced.kNN_predict(10);
    auto avg = allMetricsAveraged(reduced.getTestLabels(), predicted_labels);
    std::cout << "accuracy: " << std::get<0>(avg) << ", recall: " << std::get<1>(avg) << ", precision: " << std::get<2>(avg) << std::endl;
}
*/
/*
TEST_F (datasetTest, reducedPca) {
    reduced.trainPca(3, 0.0001);
    std::cout << reduced.getPcaVecs() << std::endl;
    std::cout << reduced.getPcaLambdas() << std::endl;

}
*/

TEST_F (datasetTest, bigPca) {
    big.trainPca(15, 0.0001);
    //std::cout << big.getPcaVecs() << std::endl;
    std::cout << big.getPcaLambdas() << std::endl;

}
/*
TEST_F (datasetTest, reducedPcaKnn) {
    reduced.shuffle();
    //reduced.splitTrainFromTest(0.3);
    reduced.trainPca(3, 0.0001);
    reduced.pca_kNN_predict(10, 3, 0.0001);
}*/