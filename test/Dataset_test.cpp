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

    Dataset reduced = Dataset("../../test/casos_test/", "testFullRed.in");
    // Dataset big = Dataset("../../test/casos_test/", "testFullBig.in");

    Matrix l = Matrix(3,1);
    Matrix g = Matrix(3,3);
    Matrix e = Matrix(5,3);
    Dataset littleMock;
};
/*
TEST_F (datasetTest, lab) {


    std::vector<int> a { 1,2,3 },
            b {9,0,-7};

    std::vector<int> c;
    c.insert(end(c),begin(a), end(a));
    c.insert(end(c), begin(b), end(b));

    for(auto it = c.begin(); it != c.end(); it++) {
        std::cout << *it << std::endl;
    }

    // big.knnEquitativeSamplingKFold(5);
    // auto scores_per_fold = reduced.;
    // std::cout << std::get<0>(scores_per_fold[0]) << std::endl;

}



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
    auto avg = allMetricsWrapper(reduced.getTestLabels(), predicted_labels);
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
/*
TEST_F (datasetTest, bigPca) {
    //big.trainPca(15, 0.0001);
    std::cout << big.getPcaVecs() << std::endl;
    std::cout << big.getPcaLambdas() << std::endl;
    std::cout << big.getTrainImages() << std::endl;
    std::cout << big.getTrainImages().cols() << std::endl;
}
*/
TEST_F (datasetTest, reducedPcaKnn) {

    int k = 0;
    int amount_of_people = 41;
    int picks_per_person = 10;
    auto imageFold = reduced.getEquitativeSamplingFold(reduced.getTrainImages(), k, amount_of_people, picks_per_person);
    auto labelFold = reduced.getEquitativeSamplingFold(reduced.getTrainLabels(), k, amount_of_people, picks_per_person);

    Dataset d = Dataset(std::get<0>(imageFold), std::get<0>(labelFold),
                        std::get<1>(imageFold), std::get<1>(labelFold));
    d.trainPca(10, 0.0001);
    allMetricsWrapper(std::get<1>(labelFold),d.pca_kNN_predict(3, 0.0001));

}