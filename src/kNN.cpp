#include "kNN.h"
#include <algorithm>




Matrix distance(const Matrix& data, const Matrix& image) { /* returns a column Matrix with the distance
 of every image on data with the one that enters as a parameter //The image is taken as a col vector*/
    assert(data.cols() == image.rows());
    Matrix res(data.rows(),1);
    std::cout<<"res matrix"<< endl << res;
    std::cout<<"data matrix"<< endl << data;
    std::cout<<"image matrix"<< endl <<image;
    Matrix aux = Matrix(data.rows(),image.rows());
    std::cout<<"aux matrix" << endl << aux;
    for (int i = 0; i < data.rows(); ++i) {
        for (int j = 0; j < image.rows(); ++j) {
            aux.setIndex(i, j, image(j));
        }
    }
    std::cout<<"aux matrix" << endl << aux <<endl;
    Matrix difference = Matrix(data.rows(),data.rows());
    difference = data-aux;
    std::cout<<"data - aux" << endl <<difference<< endl;
    for (int i = 0; i < difference.rows(); ++i){
        double sum = 0;
        for (int j = 0; j < difference.cols(); ++j) {
            int value = difference(i,j);
            std::cout <<"value is " << value << endl;
            sum = sum + (value * value);
        }
        std::cout<<"sum sub " << i <<" " << sum << endl;
        res.setIndex(i, 0, sqrt(sum)); //sets in every row of res the distance between the imput image and the
    }
    std::cout<<"distances matrix" << endl << res << endl;

    return res;
}

int mostAppears(const int array[], const int length){/* returns the index of the biggest element (the imput
 array has the number of repetitions of it's index. */
    int mostRepeated = 0;
    if (length > 1) {
        for (int i = 0; i < length; ++i) {
            if(array[i] > array[mostRepeated]){
                mostRepeated = i;
            }
        }
    }
    return mostRepeated+1;
}

bool myComparison(const tuple<int,double> a, const tuple<int,double> b) {//used to sort v

    return std::get<1>(a) < std::get<1>(b);
}

int kNN(const Matrix& data, const Matrix& image, int k, const int numberOfPeople, const int numberOfPicturesPerPeople) {
    assert(data.rows() >= k && k > 0); //the number of neighbours must be equal or less to the number of pictures on the dataset
    Matrix distances = distance(data, image);
    vector<tuple<int, double>> v; /*first element of tuple identifies the person on data image, second is the distance
    to the imput image. */
    int person = 1;
    int picCounter = 0;
    for (int i = 0; i < distances.rows(); ++i) { /* sets v to be as needed (a vector of tuples with the id of the person
        and the distance of his picture to the imput image) */
        v.push_back(std::make_tuple(person, distances(i)));
        picCounter++;
        if (picCounter == numberOfPicturesPerPeople){
            person++;
            picCounter = 0;
        }
    }

    sort(v.begin(), v.end(), myComparison); //Sorts v from the shortest distance to the largest.

    int repetitions[numberOfPeople] = {}; /*this array is used to count the number on the k nearest neighbours.
    Our data set has 41 persons, so we will count as max 41 repetitions.*/
    for (int i = 0; i < k; k++) {
        repetitions[get<0>(v[i]) - 1]++; /*ads one to the number of repetitions of the person
// * on the ith nearest position */// substraction of one is due to repetitions index from 0
    }
//    std::cout<<endl<< "number of appearences of each person on the kNN: ";
//    for (int i = 0; i < numberOfPeople; ++i) {
//        std::cout << repetitions[i]<<endl; //fixme: don't know why this returns 2147483647 instead of the elements of repetitions

//    }
        return mostAppears(repetitions, numberOfPeople); //returns the person that appears the most on the kNN.
}