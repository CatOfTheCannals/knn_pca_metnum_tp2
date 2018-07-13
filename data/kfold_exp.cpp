#include "kfold_exp.h"

void run_diff_kfold_knn() {
    std::vector<int> neighbourhood_sizes = {1,3,8,10,15,25};
    for(std::vector<int>::iterator neighbours = neighbourhood_sizes.begin(); neighbours != neighbourhood_sizes.end(); ++neighbours) {
        kfold_knn(*neighbours);
    }
}

void kfold_knn(int neighbours) {
    std::ostringstream filename;
    filename << "../../data/results/pija/kfold_knn_" << neighbours << ".csv";
    ofstream file;
    file.open(filename.str());
    file << "accuracy" << "," << "groundTruth" << "," << "prediction" << std::endl;

    Dataset d = Dataset("../../test/casos_test/", "testFullBig.in");
    std::vector<std::tuple<double, std::vector<double>, std::vector<double>>>
            scores_per_fold = d.knnEquitativeSamplingKFold(neighbours);

    for(int i = 0; i < scores_per_fold.size(); i++) {
        file << std::get<0>(scores_per_fold[i]) << "," << vecOfDoublesToString(std::get<1>(scores_per_fold[i]))
             << "," << vecOfDoublesToString(std::get<2>(scores_per_fold[i])) << std::endl;
    }

}

void run_diff_kfold_knn_pca() {
    std::vector<int> neighbourhood_sizes = {1,3,8,10,15,25};
    std::vector<int> alphas = {3,10,25,40,60};
    
    for(std::vector<int>::iterator alpha = alphas.begin(); alpha != alphas.end(); ++alpha){
	for(std::vector<int>::iterator neighbours = neighbourhood_sizes.begin(); neighbours != neighbourhood_sizes.end(); ++neighbours) {
            kfold_knn_with_pca(*neighbours, *alpha);
        }
    }
}

void kfold_knn_with_pca(int neighbours, int alpha) {
    std::ostringstream filename;
    filename << "../../data/results/knn_pca_metrics_hacked_kfold/kfold_knn_" << neighbours << "_pca_" << alpha << ".csv";
    ofstream file;
    file.open(filename.str());
    file << "accuracy" << "," << "groundTruth" << "," << "predictionx" << std::endl;

    Dataset d = Dataset("../../test/casos_test/", "testFullBig.in");
    std::vector<std::tuple<double, std::vector<double>, std::vector<double>>>
            scores_per_fold = d.pcaKnnEquitativeSamplingKFold(neighbours, alpha);

    for(int i = 0; i < scores_per_fold.size(); i++) {
        file << std::get<0>(scores_per_fold[i]) << "," << vecOfDoublesToString(std::get<1>(scores_per_fold[i]))
             << "," << vecOfDoublesToString(std::get<2>(scores_per_fold[i])) << std::endl;
    }

}
