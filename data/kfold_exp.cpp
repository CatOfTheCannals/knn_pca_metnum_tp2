#include "kfold_exp.h"

void run_diff_kfold_knn() {
    for(int neighbours = 1; neighbours < 30; neighbours++) {
        kfold_knn(neighbours);
    }
}

void kfold_knn(int neighbours) {
    std::ostringstream filename;
    filename << "../../data/results/knn_metrics/kfold_knn_" << neighbours << ".csv";
    ofstream file;
    file.open(filename.str());
    file << "accuracy" << "," << "recall_at_k" << "," << "precision_at_k" << std::endl;

    Dataset d = Dataset("../../test/casos_test/", "testRed.in");
    std::vector<std::tuple<double, std::vector<double>, std::vector<double>>>
            scores_per_fold = d.knnEquitativeSamplingKFold(neighbours);

    for(int i = 0; i < scores_per_fold.size(); i++) {
        file << std::get<0>(scores_per_fold[i]) << "," << vecOfDoublesToString(std::get<1>(scores_per_fold[i]))
             << "," << vecOfDoublesToString(std::get<2>(scores_per_fold[i])) << std::endl;
    }

}

void run_diff_kfold_knn_pca() {
    for(int alpha = 1; alpha < 411; alpha += 41) {
        for(int neighbours = 1; neighbours < 30; neighbours++) {
            kfold_knn_with_pca(neighbours, alpha);
        }
    }
}

void kfold_knn_with_pca(int neighbours, int alpha) {
    std::ostringstream filename;
    filename << "../../data/results/knn_pca_metrics/kfold_knn_" << neighbours << "_pca_" << alpha << ".csv";
    ofstream file;
    file.open(filename.str());
    file << "accuracy" << "," << "recall_at_k" << "," << "precision_at_k" << std::endl;

    Dataset d = Dataset("../../test/casos_test/", "testRed.in");
    std::vector<std::tuple<double, std::vector<double>, std::vector<double>>>
            scores_per_fold = d.pcaKnnEquitativeSamplingKFold(neighbours, alpha);

    for(int i = 0; i < scores_per_fold.size(); i++) {
        file << std::get<0>(scores_per_fold[i]) << "," << vecOfDoublesToString(std::get<1>(scores_per_fold[i]))
             << "," << vecOfDoublesToString(std::get<2>(scores_per_fold[i])) << std::endl;
    }

}