#include "Svd.hpp"


tuple<Matrix, double> power_method(Matrix& x_0, Matrix& input,
                                   double epsilon) {
    assert(input.rows() == input.cols());
    assert(input.rows() == x_0.rows());
    assert(1 == x_0.cols());

    Matrix x(x_0 / x_0.norm());
    Matrix A(input);

    double norm_err = 0;
    double delta_norm_err = 0;
    Matrix last_Ax(input.rows(), input.cols());

    do {
        auto Ax = A.multiply(x);
        last_Ax = Ax;
        auto Ax_norm = Ax.norm();

        // en el caso que la matrix tenga el autovector cero
        if (Ax_norm < epsilon) {
            return make_tuple(x, (double)0);
        }

        auto curr_x = Ax / Ax_norm;
        auto err = x - curr_x;
        x = curr_x;

        // guardo la norma del error del paso anterior
        auto prev_norm_err = norm_err;
        norm_err = err.norm();
        delta_norm_err = prev_norm_err - norm_err;

        // tomo el valor absoluto
        delta_norm_err = delta_norm_err > 0 ? delta_norm_err : -delta_norm_err;
    } while (norm_err > epsilon && delta_norm_err > epsilon);

    auto lambda = x.transpose().multiply(last_Ax);
    return make_tuple(x, lambda(0));
}

tuple<Matrix, Matrix> svd(const Matrix &A, unsigned int num_components,
                          double epsilon) {
    cout << "Running "
         << "SVD(num_components=" << num_components << "). Input: " << A.rows()
         << " rows" << endl;

    auto start = chrono::steady_clock::now();

    Matrix x_0( ones(A.rows(), 1));

/*
    Matrix X(A);

    assert((num_components <= X.rows()) && (num_components <= X.cols()));

    Matrix lambdas(num_components, num_components);
    Matrix k_eigen_vectors(X.rows(), num_components);

    for (auto i = 0; i < num_components; i++) {

        Matrix eigen_vector(A.rows(), 1);
        double eigen_value;
        tie(eigen_vector, eigen_value) =
                power_method(x_0, X, epsilon); // calculate i_th eigen vector and it's value

        for (auto q = 0; q < k_eigen_vectors.rows(); q++) {
            k_eigen_vectors(q, i) =
                    eigen_vector(q, 0); // fill eigen vector in res matrix
        }
        lambdas(i, i) = eigen_value;

        auto external = eigen_vector * eigen_vector.transpose();
        auto internal = eigen_vector.transpose() * eigen_vector;
        double internal_val = internal(0, 0);

        X = X - (external * (eigen_value / internal_val));

        if ((i + 1) % 10 == 0) {
            cout << "SVD(num_components=" << num_components << "): " << i + 1 << "/"
                 << num_components << " done" << endl;
        }
    }

    cout << "SVD(num_components=" << num_components << "): " << num_components
         << "/" << num_components << " done" << endl;

    auto end = chrono::steady_clock::now();
    auto delta_t = chrono::duration<double, milli>(end - start).count();
    cout << "Run SVD(num_components=" << num_components << ") in "
         << (delta_t / 1000.0) / 60.0 << " mins" << endl;

    return make_tuple(k_eigen_vectors, lambdas);
    */
}

Matrix ones(int i, int j)  {
    Matrix res(i, j);
    for (std::size_t i = 0; i < i; i++) {
        for (std::size_t j = 0; j < j; j++) {
            res.setIndex(i, j, 1);
        }
    }
    return res;
}