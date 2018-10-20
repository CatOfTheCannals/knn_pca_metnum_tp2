#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <string>
#include <string.h>

#include "src/Dataset.h"

int main(int argc, char** argv){

    std::string train_path(".");
    std::string test_path(".");


    Dataset::loadImdbVectorizedReviews();




    return 0;

}