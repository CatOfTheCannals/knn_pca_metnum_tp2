#include "kNN.hpp"
#include <math.h>
#include <tuple>
#include <vector>
#include <algorithm>
#include <iostream>




Matrix distance(Matrix& data, Matrix& image) { /* returns a column Matrix with the distance
 of every image on data with the one that enters as a parameter */
    assert(data.cols() == image.cols());
    Matrix res(data.rows(),1);

    Matrix aux = Matrix(data.rows(),data.cols());
    for (int i = 1; i <= data.rows(); ++i) {
        aux.setIndex(i, 1, 1);
    }
    aux.multiply(image); //creates a matrix with image in all it's rows.
    data-aux;
    for (int i = 1; i <= aux.rows(); ++i){
        double sum = 0;
        for (int j = 1; j <= aux.cols(); ++j) {
            int value = aux(i,j);
            sum = sum + value * value;
        }
        res.setIndex(i,1, sqrt(sum)); //sets in every row of res the distance between the imput image and the
    }
    return res;
}
//
//int mostAppears(int array[], int length){//fixme: check the parameters
//    int mostRepeated = 0;
//    if (length > 1) {
//        for (int i = 0, i < length; ++i) {
//            if(array[i] > array[mostRepeated]){
//                mostRepeated = i;
//            }
//        }
//    }
//    return mostRepeated+1;
//}
//
//
//int kNN(Matrix& data, Matrix& image, int k, int numberOfPeople) {
//
//    Matrix distance = distance(data, image);
//    vector<tuple<int, double>> v; /*first element of tuple identifies the person on data image, second is the distance
//    to the imput image. */
//    int person = 1;
//    for (int i = 0; i < distance.rows(); i++) { //sets v to be as needed.
//        if (i - person * 10 != 0) { //used to diferenciate between persons
//            v.push_back(std::make_tuple(person, distance(i))) //fixme: suposing that () gives you the ith element
//        } else {
//            v.push_back(std::make_tuple(person, distance(i))) //fixme: suposing that () gives you the ith element
//            person++;
//        }
//    }
//    bool myComparison(const pair<int,double> &a,const pair<int,double> &b) //used to sort v
//    {
//        return a.second<b.second;
//    }
//    sort(v.begin(),v.end(),myComparison); //Sorts v from the shortest distance to the largest.
//    int repetitions [numberOfPeople] = { }; /*this array is used to count the number on the k nearest neighbours.
//    Our data set has 41 persons, so we will count as max 41 repetitions.*/
//    for(int i = 0; i < k ; k++) {
//        repetitions[get<0>(v[i])-1]++; /*ads one to the number of repetitions of the person
// * on the ith nearest position */// substraction of one is due to repetitions index from 0
//    }
//    return mostAppears(repetitions, numberOfPeople); //returns the person that appears the most on the kNN.
//}