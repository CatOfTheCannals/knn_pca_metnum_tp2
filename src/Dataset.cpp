#include "Dataset.h"

Matrix Dataset::getTestImages() const{
    Matrix output(_testImages);
    return output;
}

Matrix Dataset::getTestLabels() const{
    Matrix output(_testLabels);
    return output;
}

Matrix Dataset::getTrainImages() const{
    Matrix output(_trainImages);
    return output;
}

Matrix Dataset::getTrainLabels() const{
    Matrix output(_trainLabels);
    return output;
}

Matrix Dataset::getPcaVecs() const {
    Matrix output(_pcaVecs);
    return output;
}
Matrix Dataset::getPcaLambdas() const {
    Matrix output(_pcaLambdas);
    return output;
}

void Dataset::shuffle() {
    int n = _trainImages.rows();
    srand (time(NULL));
    for(int i = 0; i < n; i++) {
        int swap_index = rand() % (n - i) ;
        _trainImages.swapRows(i, i + swap_index);
        _trainLabels.swapRows(i, i + swap_index);
    }
}

Matrix& Dataset::get_mt_times_m() {
    if(this->mt_times_m_is_set == 0){
        generate_mt_times_m();
        this->mt_times_m_is_set = 1;
    }
    return this->_mt_times_m;
}

void Dataset::generate_mt_times_m(){
    Matrix X(_trainImages);
    Matrix mean(1, _trainImages.cols());
    // get "mean row"
    for(int i = 0; i < _trainImages.rows(); i++){
        mean = mean + (X.getRow(i) / X.rows());
    }
    // subtract mean row to every row: center them
    for(int i = 0; i < X.rows(); i++){
        auto centered_row = X.getRow(i) - mean;
        centered_row = centered_row / sqrt(X.rows() - 1);
        for (int j = 0; j < X.cols(); ++j) {
            X.setIndex(i,j, centered_row(j));
        }
    }

    //Matrix trans = X.transpose();
    this->_mt_times_m = X.mt_times_m();

}
void Dataset::splitTrainFromTest(double testPercentage) {
    int testRows = _trainImages.rows() * testPercentage;
    cout << " test rows: "<< testRows << " rows: "<< _trainImages.rows() << endl;
    _testImages = _trainImages.subMatrix(0, testRows, 0, _trainImages.cols() - 1);
    _testLabels = _trainLabels.subMatrix(0, testRows, 0, _trainLabels.cols() - 1);
    _trainImages = _trainImages.subMatrix(testRows + 1, _trainImages.rows() - 1, 0, _trainImages.cols() - 1);
    _trainLabels = _trainLabels.subMatrix(testRows + 1, _trainLabels.rows() - 1, 0, _trainLabels.cols() - 1);
}

void Dataset::trainPca(int alpha, double epsilon) {
    assert(_trainImages.rows() > 0);
    Matrix M = get_mt_times_m();
    auto pca_eigenvectors_and_eigenvalues = svd(M, alpha, epsilon);
    _pcaVecs = std::get<0>(pca_eigenvectors_and_eigenvalues);
    _pcaAlpha = alpha;
    _pcaLambdas = std::get<1>(pca_eigenvectors_and_eigenvalues);
    _transformedTrainImages = _trainImages*_pcaVecs;
    std::cout << "_pcaVecs " << _pcaVecs.rows() << ", " << _pcaVecs.cols() << std::endl;
    std::cout << "_transformedTrainImages " << _transformedTrainImages.rows() << ", " << _transformedTrainImages.cols() << std::endl;

}

Matrix Dataset::pca_kNN_predict_old(int k) const {

    Matrix testLabels = Matrix(_testImages.rows(), 1);
    // std::cout << "rows " << _testImages.rows() << std::endl;
    for(int i = 0; i < _testImages.rows(); i++) {
        auto begin = GET_TIME;
        // std::cout << "entro " << i << std::endl;
        auto characteristic_transformation = _testImages.getRow(i)*_pcaVecs;
        // characteristic_transformation.show_matrix();
        // std::cout << "char trans ok " << std::endl;
        int ith_label = kNN(_transformedTrainImages, _trainLabels, characteristic_transformation, k);
        // std::cout << "knn pred  ok " << std::endl;
        testLabels.setIndex(i , 0 ,ith_label);
        // std::cout << "set index ok " << std::endl;
        auto end = GET_TIME;
        //if(i%100==0){cout << "i: "<< i <<" time: "<< 100*GET_TIME_DELTA(begin, end)<< endl;}
    }
    std::cout << " salio del loop" << std::endl;
    return testLabels;
}

Matrix Dataset::pca_kNN_predict_new(int k, int alpha) {
    assert(alpha <= _pcaAlpha);
    Matrix testLabels = Matrix(_testImages.rows(), 1);
    cout << "retraining model for new alpha: "<< alpha << endl;
    //_transformedTrainImages = trainImages*_pcaVecs
    _transformedTrainImages = Matrix(_trainImages.rows(),alpha);
    for(int i = 0; i < _trainImages.rows(); i++) {
        auto begin = GET_TIME;
        for(int j = 0; j < alpha; j++){
            double acum = 0.0;
            for(int k = 0; k < _pcaVecs.rows() ; k++){
                acum+=_testImages(i,k)*_pcaVecs(k,j);
                }
            _transformedTrainImages.setIndex(i,j,acum);
        }
        auto end = GET_TIME;
        if(i%100==0){cout << "new train i: "<< i <<" time: "<< 100*GET_TIME_DELTA(begin, end)<< endl;}
    }
    cout << "done training model for alpha: "<< alpha << endl;
    // iterate through test instances
    for(int i = 0; i < _testImages.rows(); i++) {
        auto begin = GET_TIME;

        //generate the characteristic transform
        Matrix characteristic_transformation = Matrix(1, _transformedTrainImages.cols());
        for(int j = 0; j < alpha; j++){
            double acum = 0.0;
            for(int k = 0; k < _pcaVecs.rows() ; k++){
                acum+=_testImages(i,k)*_pcaVecs(k,j);
                }
            characteristic_transformation.setIndex(0,j,acum);
        }


        // knn predict
        int ith_label = kNN(_transformedTrainImages, _trainLabels, characteristic_transformation, k);
        testLabels.setIndex(i , 0 ,ith_label);
        auto end = GET_TIME;
    }
    return testLabels;
}


Matrix Dataset::kNN_predict(int k) const {
    Matrix testLabels = Matrix(_testImages.rows(), 1);
    for(int i = 0; i < _testImages.rows(); i++) {
        std::cout << "call knn for row: " << i << std::endl;
        int ith_label = kNN(_trainImages, _trainLabels, _testImages.getRow(i), k);
        testLabels.setIndex(i ,0 , ith_label);
    }
    return testLabels;
}
std::vector<std::tuple<double, std::vector<double>, std::vector<double>>>
Dataset::knnEquitativeSamplingKFold(int neighbours, bool bigTestSet = false)  {

    int amount_of_people = 41;
    int amount_of_picks = _trainImages.rows();
    int picks_per_person = amount_of_picks / amount_of_people;
    std::vector<std::tuple<double, std::vector<double>, std::vector<double>>>
        scores_per_fold;
    shuffleSamePersonPicks(amount_of_people, picks_per_person);

    for(int k = 0; k < 5; k++) {
        auto imageFold = getEquitativeSamplingFold(_trainImages, k, amount_of_people, picks_per_person, bigTestSet);
        auto labelFold = getEquitativeSamplingFold(_trainLabels, k, amount_of_people, picks_per_person, bigTestSet);

        Dataset d = Dataset(std::get<0>(imageFold), std::get<0>(labelFold),
                            std::get<1>(imageFold), std::get<1>(labelFold));


        scores_per_fold.push_back(allMetricsWrapper(std::get<1>(labelFold), d.kNN_predict(neighbours)));

    }

    return scores_per_fold;
}
std::vector<std::tuple<double, std::vector<double>, std::vector<double>>>
Dataset::pcaKnnEquitativeSamplingKFold(int neighbours, int alpha, bool bigTestSet = false) {

    double epsilon = 0.0001;
    int amount_of_people = 41;
    int amount_of_picks = _trainImages.rows();
    int picks_per_person = amount_of_picks / amount_of_people;
    std::vector<std::tuple<double, std::vector<double>, std::vector<double>>>
            scores_per_fold;

    shuffleSamePersonPicks(amount_of_people, picks_per_person);

    for(int k = 0; k < 5; k++) {
        auto imageFold = getEquitativeSamplingFold(_trainImages, k, amount_of_people, picks_per_person, bigTestSet);
        auto labelFold = getEquitativeSamplingFold(_trainLabels, k, amount_of_people, picks_per_person, bigTestSet);

        Dataset d = Dataset(std::get<0>(imageFold), std::get<0>(labelFold),
                            std::get<1>(imageFold), std::get<1>(labelFold));
        std::cout << "train_pca" << std::endl;
        d.trainPca(alpha, epsilon);
        scores_per_fold.push_back(allMetricsWrapper(std::get<1>(labelFold),
                                                    d.pca_kNN_predict_old(neighbours)));
    }

    return scores_per_fold;
}

void Dataset::shuffleSamePersonPicks(int amount_of_people, int picks_per_person) {

    srand (time(NULL));

    for(int person = 0; person < amount_of_people; person++) {
        for(int i = 0; i < picks_per_person; i++) {
            int swap_index = rand() % (picks_per_person - i) ;
            _trainImages.swapRows(i + person * picks_per_person, i + person * picks_per_person + swap_index);
            _trainLabels.swapRows(i + person * picks_per_person, i + person * picks_per_person + swap_index);
        }
    }

}


std::tuple<Matrix, Matrix> Dataset::getEquitativeSamplingFold
        (const Matrix &input_matrix, int iteration, int amount_of_people, int picks_per_person, bool bigTestSet = false) const {

    /* distribute dataset
     * it is assumed that the dataset will have labels increasingly ordered
     */
    std::vector<std::vector<Matrix>> personBuckets;
    int amountOfPersons = 41;
    int picksPerPerson = 10;
    for(int person = 0; person < amountOfPersons ; person++) {
        std::vector<Matrix> bucket;
        for(int pick = 0; pick < picksPerPerson; pick++) {
            bucket.push_back(input_matrix.getRow(person * picksPerPerson + pick));
        }
        personBuckets.push_back(bucket);
    }

    // separe test from train
    std::vector<Matrix> testSamplesVec;
    for(int person = 0; person < amountOfPersons ; person++) {
        testSamplesVec.push_back(personBuckets[person][iteration]);
        testSamplesVec.push_back(personBuckets[person][iteration+1]);
        personBuckets[person].erase(personBuckets[person].begin() + iteration + 1);
        personBuckets[person].erase(personBuckets[person].begin() + iteration);
    }

    // flatten the remaining rows from personBuckets
    std::vector<Matrix> trainSamplesVec;
    for(auto it = personBuckets.begin(); it != personBuckets.end(); it++) {
        auto bucket = *it;
        trainSamplesVec.insert(end(trainSamplesVec),begin(bucket), end(bucket));
    }

    auto trainSamples = Matrix::vecOfRowsToMatrix(trainSamplesVec);
    auto testSamples = Matrix::vecOfRowsToMatrix(testSamplesVec);

    if(bigTestSet){
        return std::make_tuple(testSamples, trainSamples);
    } else {
        return std::make_tuple(trainSamples, testSamples);
    }


}

Dataset Dataset::loadImdbVectorizedReviews() {
    auto filter_out = [] (const int token, const FrecuencyVocabularyMap & vocabulary) {
        double token_frecuency = vocabulary.at(token);
        return token_frecuency < 0.01 || token_frecuency > 0.99;
    };
    auto train_and_test_vectorized_matrices_and_labels = build_vectorized_datasets(filter_out);
    auto train_vectorized_matrix_and_label = std::get<0>(train_and_test_vectorized_matrices_and_labels);
    auto test_vectorized_matrix_and_label = std::get<1>(train_and_test_vectorized_matrices_and_labels);
    
    auto train_vector_matrix = std::get<0>(train_vectorized_matrix_and_label);
    auto test_vector_matrix = std::get<0>(test_vectorized_matrix_and_label);

    auto train_vector_label = std::get<1>(train_vectorized_matrix_and_label);
    auto test_vector_label = std::get<1>(test_vectorized_matrix_and_label);

    return Dataset(train_vector_matrix, train_vector_label,
                   test_vector_matrix, test_vector_label);


}