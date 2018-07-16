#include "pca_cuantitative.h"
#include "knn_cuantitative.h"
#include "pca_knn_cuantitative.h"
#include "kfold_exp.h"

int main(){
    std::cout << "Starting exp..." << std::endl << std::endl;
    // pca_cuantitative();
    // knn_cuantitative();
    // pca_knn_cuantitative();
     // kfold_knn(1);
    //kfold_knn_with_pca(5, 15);
    // run_diff_kfold_knn_pca();
    // run_diff_kfold_knn();



    Dataset d = Dataset("/Users/cgiudice/metnum/knn_pca_metnum_tp2/test/casos_test/", "testRed.in",
                        "/Users/cgiudice/metnum/knn_pca_metnum_tp2/test/casos_test/", "testRed.in");

    int method = 0;
    Matrix predictions;
    if(method == 1) {
        predictions = d.pca_kNN_predict(1, 25);
        std::cout << "k: 1, alpha: 25" << std::endl;
    } else {
        predictions = d.kNN_predict(1);
        std::cout << "k: 1" << std::endl;
    }

    allMetricsWrapper(d.getTestLabels(), predictions);

    std::cout << std::endl << "Finished exp." << std::endl;
}
