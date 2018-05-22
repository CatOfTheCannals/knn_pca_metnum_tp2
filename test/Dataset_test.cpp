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

    }
    Dataset reduced = Dataset("../../test/casos_test/", "testBig.in");

};

// este test tarda banda en correr, agrandar el epsilon o disminuir max iters
TEST_F (datasetTest, pcaReducedImages){
    std::cout << reduced.getImages().rows() << " " <<    reduced.getImages().cols() << std::endl;

    auto begin = GET_TIME;
    auto pca_eigenvectors_and_eigenvalues = pca(reduced.getImages(), 15, 0.00001);
    auto end = GET_TIME;

    Matrix autoVecs(std::get<0>(pca_eigenvectors_and_eigenvalues));
    Matrix lambdas(std::get<1>(pca_eigenvectors_and_eigenvalues));

    std::cout << "EigenVectors:" << std::endl << autoVecs << std::endl;
    std::cout << "EigenValues:" << std::endl << lambdas << std::endl;
    std::cout << "Time elapsed: " << GET_TIME_DELTA(begin, end) << std::endl;
}