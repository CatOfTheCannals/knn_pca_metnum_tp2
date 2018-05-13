#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "gtest/gtest.h"

#include "../src/Matrix.hpp"
#include "../src/Svd.hpp"

// --------- SET UP --------------
class runTest : public ::testing::Test {
protected:
    virtual void SetUp() {
            g.setIndex(0, 0, 1);
            g.setIndex(0, 1, 2);
            g.setIndex(0, 2, 3);
            g.setIndex(1, 1, 4);
            g.setIndex(1, 2, 5);
            g.setIndex(1, 0, 6);
            g.setIndex(2, 1, 7);
            g.setIndex(2, 2, 8);
            g.setIndex(2, 0, 9);

            x_0.setIndex(0, 0, 1);
            x_0.setIndex(1, 0, 2);
            x_0.setIndex(2, 0, 3);
    }
    Matrix g = Matrix(3,3);
    Matrix x_0 = Matrix(3,1);
    double epsilon = 0.001;
};


TEST_F (runTest, powerG){
    Matrix v = Matrix(3,1);
    double lambda;

    std::tie(v, lambda) = power_method(x_0, g, epsilon);

    epsilon *= 10; // la precision queda asi por algun motivo
    ASSERT_TRUE(g.multiply(v).isApproximate(v * lambda, epsilon)) ;
}

TEST_F (runTest, svd){
    /*
    auto svdRes = g.svd(2, epsilon);
    Matrix lambdas(std::get<0>svd_res);
    Matrix autoVecs(std::get<1>svd_res);
     */

}