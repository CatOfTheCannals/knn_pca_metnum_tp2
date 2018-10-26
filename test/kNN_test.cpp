#include <iostream>
#include "gtest/gtest.h"

#include "../src/Matrix.h"
#include "../src/kNN.h"

// --------- SET UP --------------
class runTest : public ::testing::Test {
protected:
    virtual void SetUp() {

        image_1.setIndex(0, 0, 0);
        image_1.setIndex(0, 1, 0);
    }
    Matrix image_1 = Matrix(1,2);

};



TEST_F (runTest, trainSetSizeOneReturnsSameLabel){

    Matrix ts_size_one = Matrix(1,2);
    ts_size_one.setIndex(0, 0, 1);
    ts_size_one.setIndex(0, 1, 2);

    Matrix positive_label_size_one = Matrix(1,1);
    positive_label_size_one .setIndex(0,0,1);
    Matrix negative_label_size_one = Matrix(1,1);
    negative_label_size_one.setIndex(0,0,0);

    ASSERT_EQ( kNN(ts_size_one, positive_label_size_one,image_1, 1), 1);
    ASSERT_EQ( kNN(ts_size_one, negative_label_size_one,image_1, 1), 0);
}

TEST_F (runTest, trainSetSizeTwoReturnsNearestLabel){

    Matrix ts_size_two = Matrix(2,2);
    ts_size_two.setIndex(0, 0, 1);
    ts_size_two.setIndex(0, 1, 2);
    ts_size_two.setIndex(1, 0, 2);
    ts_size_two.setIndex(1, 1, 2);

    Matrix label_size_two = Matrix(2,1);
    label_size_two.setIndex(0,0,0);
    label_size_two.setIndex(1,0,1);

    ASSERT_EQ( kNN(ts_size_two, label_size_two,image_1, 1), 0);
}

TEST_F (runTest, kThreeHasTwoPositives){

    Matrix ts_size_four = Matrix(4,2);
    ts_size_four.setIndex(0, 0, 1);
    ts_size_four.setIndex(0, 1, 2);
    ts_size_four.setIndex(1, 0, 2);
    ts_size_four.setIndex(1, 1, 2);
    ts_size_four.setIndex(2, 0, 3);
    ts_size_four.setIndex(2, 1, 2);
    ts_size_four.setIndex(3, 0, 4);
    ts_size_four.setIndex(3, 1, 2);

    Matrix label_size_four = Matrix(4,1);
    label_size_four.setIndex(0,0,0);
    label_size_four.setIndex(1,0,1);
    label_size_four.setIndex(2,0,1);
    label_size_four.setIndex(3,0,0);

    ASSERT_EQ( kNN(ts_size_four, label_size_four,image_1, 3), 1);
}

TEST_F (runTest, kThreeHasTwonegatives){

    Matrix ts_size_four = Matrix(4,2);
    ts_size_four.setIndex(0, 0, 1);
    ts_size_four.setIndex(0, 1, 2);
    ts_size_four.setIndex(1, 0, 2);
    ts_size_four.setIndex(1, 1, 2);
    ts_size_four.setIndex(2, 0, 3);
    ts_size_four.setIndex(2, 1, 2);
    ts_size_four.setIndex(3, 0, 4);
    ts_size_four.setIndex(3, 1, 2);

    Matrix label_size_four = Matrix(4,1);
    label_size_four.setIndex(0,0,0);
    label_size_four.setIndex(1,0,1);
    label_size_four.setIndex(2,0,0);
    label_size_four.setIndex(3,0,1);

    ASSERT_EQ( kNN(ts_size_four, label_size_four,image_1, 3), 0);
}