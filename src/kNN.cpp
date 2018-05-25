#include "kNN.h"
#include <algorithm>

Matrix distance(const Matrix& data, const Matrix& image) { /* returns a column Matrix with the distance
 of every image on data with the one that enters as a parameter //The image is taken as a row vector*/
    assert(data.cols() == image.cols());
    Matrix res(data.rows(),1);
    Matrix aux = Matrix(data.rows(),image.cols());
    for (int i = 0; i < data.rows(); ++i) {
        for (int j = 0; j < image.cols(); ++j) {
            aux.setIndex(i, j, image(j));
        }
    }
    Matrix difference = Matrix(data.rows(),data.rows());
    difference = data-aux;
    for (int i = 0; i < difference.rows(); ++i) {
        double sum = 0;
        for (int j = 0; j < difference.cols(); ++j) {
            int value = difference(i, j);
            sum = sum + (value * value);
        }
        res.setIndex(i, 0, sqrt(sum)); //sets in every row of res the distance between the imput image and the
    }
    return res;
}

int mostAppears(const vector<int> repetitions){/* returns the index of the biggest element (the imput
 array has the number of repetitions of it's index. */
    int mostRepeated = 0;
    if (repetitions.size() > 1) {
        for (int i = 0; i < (int)(repetitions.size()); ++i) {
            if(repetitions[i] > repetitions[mostRepeated]){
                mostRepeated = i;
            }
        }
    }
    return mostRepeated+1;
}

bool shortestDistance(const tuple<int,double> a, const tuple<int,double> b) {//used to sort v

    return std::get<1>(a) < std::get<1>(b);
}


bool orderedByIndex(const tuple<int,double> a, const tuple<int,double> b) {//used to sort v

    return std::get<0>(a) < std::get<0>(b);
}

int kNN(const Matrix& data, const Matrix& index, const Matrix& image, int k){
    assert(data.rows() >= k && k > 0); //the number of neighbours must be equal or less to the number of pictures on the dataset
    assert(data.rows() == index.rows());
    assert(data.rows() > 0);
    Matrix distances = distance(data, image);
    vector<tuple<int, double>> personDistances; /*first element of tuple identifies the person on data image, second is the distance
    to the imput image. */
    for (int i = 0; i < distances.rows(); ++i) { /* sets personDistances to be as needed (a vector of tuples with the id of the person
        and the distance of his picture to the imput image) */
        personDistances.push_back(std::make_tuple(index(i), distances(i)));
    }
    sort(personDistances.begin(), personDistances.end(), orderedByIndex); //Sorts personDistances to count how many different persons are on the dataset.
    int numberOfPeople = 1;
    if (personDistances.size() > 1) { //counts how many people are on the dataset that is been used.
        for (int i = 0; i < (int)(personDistances.size()) - 1; ++i) {
            if (std::get<0>(personDistances[i]) != std::get<0>(personDistances[i + 1])){
                numberOfPeople++;
            }
        }
    }
    sort(personDistances.begin(), personDistances.end(), shortestDistance); //Sorts personDistances from the shortest distance to the largest.
    std::vector<int> repetitions(numberOfPeople);/*this vector is used to count the number on the k nearest neighbours.
    Our data set has 41 persons, so we will count as max 41 repetitions.*/
    for (int i = 0; i < 40; ++i) {
        repetitions.push_back(0);
    }
    for (int i = 0; i < k; ++i) {
        repetitions.at(get<0>(personDistances[i]) - 1) = repetitions.at(get<0>(personDistances[i]) - 1) +1; /*ads one to the number of repetitions of the person
// * on the ith nearest position */// substraction of one is due to repetitions index from 0
    }
    return mostAppears(repetitions); //returns the person that appears the most on the kNN.
}

