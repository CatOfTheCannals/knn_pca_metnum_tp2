#include "kNN.h"
#include <algorithm>

Matrix distance_old(const Matrix& data, const Matrix& image) { /* returns a column Matrix with the distance
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
            double value = difference(i, j);
            sum = sum + (value * value);
        }
        res.setIndex(i, 0, sqrt(sum)); //sets in every row of res the distance between the imput image and the
    }
    return res;
}

Matrix distance(const Matrix& data, const Matrix& image) { 
/* returns a column Matrix with the distance
 of every image on data with the one that enters as a parameter 
 //The image is taken as a row vector*/
    
    assert(data.cols() == image.cols());
    Matrix res(data.rows(),1);

    for (int i = 0; i < data.rows(); ++i) {
        double sum = 0;
        for (int j = 0; j < data.cols(); ++j) {
            double value = data(i,j) - image(j);
            sum += (value * value);
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
            int repetitions_i = repetitions[i];
            int repetitions_mostRepeated = repetitions[mostRepeated];
            if(repetitions_i > repetitions_mostRepeated ){
                mostRepeated = i;
            }
        }
    }

    return mostRepeated;
}

bool shortestDistance(const tuple<int,double> a, const tuple<int,double> b) {//used to sort v

    return std::get<1>(a) < std::get<1>(b);
}

bool orderedByIndex(const tuple<int,double> a, const tuple<int,double> b) {//used to sort v

    return std::get<0>(a) < std::get<0>(b);
}

int kNN(const Matrix& data, const Matrix& labels, const Matrix& observation, int k){
    assert(data.rows() >= k && k > 0); //the number of neighbours must be equal to or less than the number of pictures on the dataset
    assert(data.rows() == labels.rows());
    assert(data.cols() == observation.cols());
    assert(observation.rows() == 1);

    int differentLabels = 2;

    Matrix distances = distance(data, observation); //vector de distancias

    vector<tuple<int, double>> personDistances; /*first element of tuple identifies the person on data observation, second is the distance
    to the imput observation. */
    for (int i = 0; i < distances.rows(); ++i) { /* sets personDistances to be as needed (a vector of tuples with the id of the person
        and the distance of his picture to the imput observation) */
        personDistances.push_back(std::make_tuple(labels(i), distances(i)));
    }


    sort(personDistances.begin(), personDistances.end(), shortestDistance); //Sorts personDistances from the shortest distance to the largest.

    if(LOG_KNN_DISTANCES) {
        cout << "distances per train instance: " << personDistances.size() << endl;
        cout << "label, distance" << endl;
        for (auto label_distance : personDistances) {
            int ith_closest_neighbour_label = get<0>(label_distance);
            double distance = get<1>(label_distance);
            cout << ith_closest_neighbour_label << ", " << distance << endl;
        }
    }

    vector<int> repetitions = vector<int>(differentLabels, 0);

    for (int i = 0; i < k; ++i) {
        int ith_closest_neighbour_label = get<0>(personDistances[i]);
        repetitions.at(ith_closest_neighbour_label)++;
    }

    /*
    cout << "repetitions vector: " << endl;
    cout << repetitions[0] << endl;
    cout << repetitions[1] << endl;
     */

    auto res = mostAppears(repetitions);

    return res; //returns the person that appears the most on the kNN.
}

