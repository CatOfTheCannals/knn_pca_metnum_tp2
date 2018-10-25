#include "Pca.h"


tuple<Matrix, Matrix> pca(const Matrix &A, unsigned int num_components, double epsilon) {
    Matrix X(A);
    Matrix mean(1, A.cols());

    // get "mean row"
    for(int i = 0; i < A.rows(); i++){
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
    auto M = X.mt_times_m();


    return svd(M, num_components, epsilon);

    // auto M_eigenvectors = X.transpose().multiply(m_eigenvectors);

    // std::cout << M_eigenvectors.rows() << ", " << M_eigenvectors.cols() << std::endl;
    // std::cout << X.rows() << ", " << X.cols() << std::endl;

    //std::cout << "difference between U's:" << std::endl ;
    //std::cout << big_m_eigenvectors - M_eigenvectors << std::endl;
    //std::cout << M_eigenvectors << std::endl;

    //std::cout << "BIG: " << big_M.multiply(big_m_eigenvectors) -big_m_eigenvectors.multiply(big_lambdas) << std::endl;
    //std::cout << std::endl << std::endl<< std::endl<< std::endl<< std::endl<< std::endl<< std::endl;
    //std::cout << "SMALL: " << M.multiply(m_eigenvectors) - m_eigenvectors.multiply(lambdas) << std::endl;

    // std::cout << "lambdas:" << std::endl << lambdas << std::endl;

    //return make_tuple(m_eigenvectors, lambdas);
}

tuple<Matrix, Matrix> svd(const Matrix &A, unsigned int num_components,
                          double epsilon) {

    Matrix X(A);
    Matrix A_copy(A);

    assert((num_components <= X.rows()) && (num_components <= X.cols()));

    Matrix lambdas(num_components, num_components);
    Matrix k_eigen_vectors(X.rows(), num_components);

    for (auto i = 0; i < num_components; i++) {

        Matrix x_0( random(A.rows(), 1));
        Matrix eigen_vector(A.rows(), 1);
        double eigen_value;

        tie(eigen_vector, eigen_value) =
                power_method(x_0, X, epsilon); // calculate i_th eigen vector and it's value

        for (auto q = 0; q < k_eigen_vectors.rows(); q++) {
            k_eigen_vectors.setIndex(q, i, eigen_vector(q, 0)); // fill eigen vector in res matrix
        }
        lambdas.setIndex(i, i, eigen_value);
        if(i%100==0){ cout <<"PCA i : " << i << endl;}
        auto external = eigen_vector*eigen_vector.transpose();
        X = X - (external * eigen_value);
    }

    return make_tuple(k_eigen_vectors, lambdas);
}

tuple<Matrix, double> power_method(Matrix& x_0, Matrix& input,
                                   double epsilon) {
    return power_method(x_0, input, epsilon, 100);

};

tuple<Matrix, double> power_method(Matrix& x_0, Matrix& input,
                                   double epsilon, int max_iters) {
    assert(input.rows() == input.cols());
    assert(input.rows() == x_0.rows());
    assert(1 == x_0.cols());



    Matrix x(x_0 / x_0.norm());
    Matrix A(input);

    double prev_norm = 0;

    auto Ax = A*x;

    for(int i = 0; i < max_iters; i++) {
        auto Ax_norm = Ax.norm();
        x = Ax / Ax_norm;
        Ax = A*x;
        if( fabs(Ax_norm - prev_norm) < epsilon ) {
            break;
        }
        prev_norm = Ax_norm;

    }

    auto lambda = x.transpose()*Ax;
    return make_tuple(x, lambda(0));
}

Matrix ones(int rows, int cols)  {
    Matrix res(rows, cols);
    for (std::size_t i = 0; i < rows; i++) {
        for (std::size_t j = 0; j < cols; j++) {
            res.setIndex(i, j, 1);
        }
    }
    return res;
}

Matrix random(int rows, int cols) {
    Matrix res(rows, cols);
    srand (time(NULL));
    for (std::size_t i = 0; i < rows; i++) {
        for (std::size_t j = 0; j < cols; j++) {
            res.setIndex(i, j, rand() % 100);
        }
    }
    return res;
}
