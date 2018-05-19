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
        f.setIndex(1, 0, 1);
        f.setIndex(1, 1, 1);
        f.setIndex(1, 2, 1);
        f.setIndex(1, 3, 1);
        f.setIndex(2, 0, 1);
        f.setIndex(2, 1, 1);
        f.setIndex(2, 2, 1);
        f.setIndex(2, 3, 1);
        f.setIndex(3, 0, 1);
        f.setIndex(3, 1, 1);
        f.setIndex(3, 2, 1);
        f.setIndex(3, 3, 1);

        g.setIndex(0, 0, 1);
        g.setIndex(0, 1, 2);
        g.setIndex(0, 2, 3);
        g.setIndex(0, 3, 4);
        g.setIndex(1, 0, 5);
        g.setIndex(1, 1, 6);
        g.setIndex(1, 2, 7);
        g.setIndex(1, 3, 8);
        g.setIndex(2, 0, 9);
        g.setIndex(2, 1, 10);
        g.setIndex(2, 2, 11);
        g.setIndex(2, 3, 12);
        g.setIndex(3, 0, 13);
        g.setIndex(3, 1, 14);
        g.setIndex(3, 2, 15);
        g.setIndex(3, 3, 16);

        res_0.setIndex(0, 0, 24);
        res_0.setIndex(1, 0, 16);
        res_0.setIndex(2, 0, 8);
        res_0.setIndex(3, 0, 0);

        res_1.setIndex(0, 0, 0);
        res_1.setIndex(1, 0, 8);
        res_1.setIndex(2, 0, 16);
        res_1.setIndex(3, 0, 24);

        res_2.setIndex(0, 0, 0);
        res_2.setIndex(1, 0, 0);
        res_2.setIndex(2, 0, 0);
        res_2.setIndex(3, 0, 0);

        x_0.setIndex(0, 0, 1);
        x_0.setIndex(1, 0, 1);
        x_0.setIndex(2, 0, 1);
        x_0.setIndex(3, 0, 1);

        x_1.setIndex(0, 0, 13);
        x_1.setIndex(1, 0, 14);
        x_1.setIndex(2, 0, 15);
        x_1.setIndex(3, 0, 16);

        x_2.setIndex(0, 0, 1);
        x_2.setIndex(1, 0, 2);
        x_2.setIndex(2, 0, 3);
        x_2.setIndex(3, 0, 4);

        //////////////////////////kNN data:
        data_1.setIndex(0, 0, 1);
        data_1.setIndex(0, 1, 2);
        data_1.setIndex(0, 2, 1);
        data_1.setIndex(0, 3, 3);
        data_1.setIndex(0, 4, 1);
        data_1.setIndex(0, 5, 9);

        image_1.setIndex(0, 0, 1);
        image_1.setIndex(1, 0, 3);
        image_1.setIndex(2, 0, 1);
        image_1.setIndex(3, 0, 3);
        image_1.setIndex(4, 0, 2);
        image_1.setIndex(5, 0, 9);

        data_2.setIndex(0, 0, 1);
        data_2.setIndex(0, 1, 2);
        data_2.setIndex(0, 2, 9);
        data_2.setIndex(0, 3, 7);
        data_2.setIndex(0, 4, 8);
        data_2.setIndex(0, 5, 9);
        data_2.setIndex(0, 6, 7);

        data_2.setIndex(1, 0, 1);
        data_2.setIndex(1, 1, 3);
        data_2.setIndex(1, 2, 9);
        data_2.setIndex(1, 3, 6);
        data_2.setIndex(1, 4, 7);
        data_2.setIndex(1, 5, 8);
        data_2.setIndex(1, 6, 6);

        data_2.setIndex(2, 0, 1);
        data_2.setIndex(2, 1, 2);
        data_2.setIndex(2, 2, 9);
        data_2.setIndex(2, 3, 2);
        data_2.setIndex(2, 4, 8);
        data_2.setIndex(2, 5, 9);
        data_2.setIndex(2, 6, 7);

        data_2.setIndex(3, 0, 1);
        data_2.setIndex(3, 1, 2);
        data_2.setIndex(3, 2, 9);
        data_2.setIndex(3, 3, 7);
        data_2.setIndex(3, 4, 8);
        data_2.setIndex(3, 5, 9);
        data_2.setIndex(3, 6, 10);

        data_2.setIndex(4, 0, 10);
        data_2.setIndex(4, 1, 10);
        data_2.setIndex(4, 2, 10);
        data_2.setIndex(4, 3, 9);
        data_2.setIndex(4, 4, 8);
        data_2.setIndex(4, 5, 9);
        data_2.setIndex(4, 6, 7);

        data_2.setIndex(5, 0, 10);
        data_2.setIndex(5, 1, 9);
        data_2.setIndex(5, 2, 8);
        data_2.setIndex(5, 3, 7);
        data_2.setIndex(5, 4, 6);
        data_2.setIndex(5, 5, 8);
        data_2.setIndex(5, 6, 9);

        data_2.setIndex(6, 0, 1);
        data_2.setIndex(6, 1, 2);
        data_2.setIndex(6, 2, 10);
        data_2.setIndex(6, 3, 8);
        data_2.setIndex(6, 4, 7);
        data_2.setIndex(6, 5, 9);
        data_2.setIndex(6, 6, 7);

        data_2.setIndex(7, 0, 1);
        data_2.setIndex(7, 1, 2);
        data_2.setIndex(7, 2, 8);
        data_2.setIndex(7, 3, 7);
        data_2.setIndex(7, 4, 8);
        data_2.setIndex(7, 5, 9);
        data_2.setIndex(7, 6, 7);

        data_2.setIndex(8, 0, 1);
        data_2.setIndex(8, 1, 2);
        data_2.setIndex(8, 2, 8);
        data_2.setIndex(8, 3, 7);
        data_2.setIndex(8, 4, 8);
        data_2.setIndex(8, 5, 7);
        data_2.setIndex(8, 6, 8);

        data_2.setIndex(9, 0, 1);
        data_2.setIndex(9, 1, 2);
        data_2.setIndex(9, 2, 9);
        data_2.setIndex(9, 3, 7);
        data_2.setIndex(9, 4, 8);
        data_2.setIndex(9, 5, 9);
        data_2.setIndex(9, 6, 9);

        image_2.setIndex(0, 0, 1);
        image_2.setIndex(1, 0, 2);
        image_2.setIndex(2, 0, 9);
        image_2.setIndex(3, 0, 11);
        image_2.setIndex(4, 0, 8);
        image_2.setIndex(5, 0, 9);
        image_2.setIndex(6, 0, 7);

        image_3.setIndex(0, 0, 10);
        image_3.setIndex(1, 0, 9);
        image_3.setIndex(2, 0, 8);
        image_3.setIndex(3, 0, 7);
        image_3.setIndex(4, 0, 6);
        image_3.setIndex(5, 0, 8);
        image_3.setIndex(6, 0, 9);



    }
    Matrix f = Matrix(4,4);
    Matrix g = Matrix(4,4);
    Matrix res_0 = Matrix(4,1);
    Matrix res_1 = Matrix(4,1);
    Matrix res_2 = Matrix(4,1);
    Matrix x_0 = Matrix(4,1);
    Matrix x_1 = Matrix(4,1);
    Matrix x_2 = Matrix(4,1);

    Matrix data_1 = Matrix(1,6);
    Matrix image_1 = Matrix(6,1);

    Matrix data_2 = Matrix(10,7);
    Matrix image_2 = Matrix(7,1);

    Matrix image_3 = Matrix(7,1);

    int rep1 [5] = { 2, 9, 3, 33, 4 };
    int rep2 [5] = { 2, 9, 3, 4, 3 };
    int rep3 [1] = { 6 };
    int rep4 [6] = { 2, 1, 1, 1, 2, 2 };
};


TEST_F (runTest, distance){
    ASSERT_EQ( distance(g, x_1), res_0);//Negative
    ASSERT_EQ( distance(g, x_2), res_1);//Positive
    ASSERT_EQ( distance(f, x_0), res_2); //Null matrix

}

TEST_F (runTest, mostAppears){
    ASSERT_EQ( mostAppears(rep1, 5), 4);//Negative
    ASSERT_EQ( mostAppears(rep2, 5), 2);//Negative
    ASSERT_EQ( mostAppears(rep3, 1), 1);//Negative
    ASSERT_EQ( mostAppears(rep4, 6), 1);//Negative

}

TEST_F (runTest, kNN){
    ASSERT_EQ( kNN(data_1,image_1, 1, 1, 1), 1);//data has one image, number of people = 1, number of pictures per people = 1
    ASSERT_EQ( kNN(data_2,image_2, 1, 1, 1), 1);//data has ten images, number of people = 1, number of pictures per people = 10
    ASSERT_EQ( kNN(data_2,image_3, 2, 5, 2), 1);//data has ten images, number of people = 1, number of pictures per people = 10

}
