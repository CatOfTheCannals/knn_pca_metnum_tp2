#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "gtest/gtest.h"

#include "../src/Matrix.hpp"
#include "../src/kNN.hpp"

// --------- SET UP --------------
class runTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        f.setIndex(0, 0, 1);
        f.setIndex(0, 1, 1);
        f.setIndex(0, 2, 1);
        f.setIndex(0, 3, 1);
        f.setIndex(0, 0, 1);
        f.setIndex(0, 1, 1);
        f.setIndex(0, 2, 1);
        f.setIndex(0, 3, 1);
        f.setIndex(0, 0, 1);
        f.setIndex(0, 1, 1);
        f.setIndex(0, 2, 1);
        f.setIndex(0, 3, 1);
        f.setIndex(0, 0, 1);
        f.setIndex(0, 1, 1);
        f.setIndex(0, 2, 1);
        f.setIndex(0, 3, 1);

        g.setIndex(0, 0, 1);
        g.setIndex(0, 1, 2);
        g.setIndex(0, 2, 3);
        g.setIndex(0, 3, 4);
        g.setIndex(0, 0, 5);
        g.setIndex(0, 1, 6);
        g.setIndex(0, 2, 7);
        g.setIndex(0, 3, 8);
        g.setIndex(0, 0, 9);
        g.setIndex(0, 1, 10);
        g.setIndex(0, 2, 11);
        g.setIndex(0, 3, 12);
        g.setIndex(0, 0, 13);
        g.setIndex(0, 1, 14);
        g.setIndex(0, 2, 15);
        g.setIndex(0, 3, 16);

        x_0.setIndex(0, 0, 1);
        x_0.setIndex(1, 0, 1);
        x_0.setIndex(2, 0, 1);
        x_0.setIndex(3, 0, 1);


        r1.setIndex(0, 0, 0);
        r1.setIndex(0, 1, 0);
        r1.setIndex(0, 2, 0);
        r1.setIndex(0, 3, 0);
        r1.setIndex(0, 0, 0);
        r1.setIndex(0, 1, 0);
        r1.setIndex(0, 3, 0);
        r1.setIndex(0, 0, 0);
        r1.setIndex(0, 2, 0);
        r1.setIndex(0, 1, 0);
        r1.setIndex(0, 2, 0);
        r1.setIndex(0, 3, 0);
        r1.setIndex(0, 0, 0);
        r1.setIndex(0, 1, 0);
        r1.setIndex(0, 2, 0);
        r1.setIndex(0, 3, 0);

        r2.setIndex(0, 0, 0);
        r2.setIndex(0, 1, 0);
        r2.setIndex(0, 2, 0);
        r2.setIndex(0, 3, 0);
        r2.setIndex(0, 0, 4);
        r2.setIndex(0, 1, 4);
        r2.setIndex(0, 3, 4);
        r2.setIndex(0, 0, 4);
        r2.setIndex(0, 2, 8);
        r2.setIndex(0, 1, 8);
        r2.setIndex(0, 2, 8);
        r2.setIndex(0, 3, 8);
        r2.setIndex(0, 0, 12);
        r2.setIndex(0, 1, 12);
        r2.setIndex(0, 2, 12);
        r2.setIndex(0, 3, 12);

        r3.setIndex(0, 0, -12);
        r3.setIndex(0, 1, -12);
        r3.setIndex(0, 2, -12);
        r3.setIndex(0, 3, -12);
        r3.setIndex(0, 0, -8);
        r3.setIndex(0, 1, -8);
        r3.setIndex(0, 3, -8);
        r3.setIndex(0, 0, -8);
        r3.setIndex(0, 2, -4);
        r3.setIndex(0, 1, -4);
        r3.setIndex(0, 2, -4);
        r3.setIndex(0, 3, -4);
        r3.setIndex(0, 0, 0);
        r3.setIndex(0, 1, 0);
        r3.setIndex(0, 2, 0);
        r3.setIndex(0, 3, 0);

        x_1.setIndex(0, 0, 13);
        x_1.setIndex(1, 0, 14);
        x_1.setIndex(2, 0, 15);
        x_1.setIndex(3, 0, 16);

        x_2.setIndex(0, 0, 1);
        x_2.setIndex(1, 0, 2);
        x_2.setIndex(2, 0, 3);
        x_2.setIndex(3, 0, 4);


    }
    Matrix f = Matrix(4,4);
    Matrix g = Matrix(4,4);
    Matrix r1 = Matrix(4,4);
    Matrix r2 = Matrix(4,4);
    Matrix r3 = Matrix(4,4);
    Matrix x_0 = Matrix(4,1);
    Matrix x_1 = Matrix(4,1);
    Matrix x_2 = Matrix(4,1);
};


TEST_F (runTest, distance){
    ASSERT_EQ( distance(g, x_1), r3);//Negative
    ASSERT_EQ( distance(g, x_2), r2);//Positive
    ASSERT_EQ( distance(f, x_0), r1); //Null matrix

}
