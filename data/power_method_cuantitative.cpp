#include "power_method_cuantitative.h"

void power_method_cuantitative(int n, double max_epsilon_ceros , int max_reps, int rep_step) {

    std::stringstream filename;
    filename << "../../data/results/power_method_cuantitative/"
                      << "n_" << n << "_max_epsilon_ceros_" << max_epsilon_ceros
                      << "_max_reps_" << max_reps << "_rep_step_" << rep_step << ".csv";
    std::cout << filename.str() << std::endl;
    ofstream file(filename.str());

    file << "n" << "," << "epsilon" << "," << "reps" << "time" << "mse" << std::endl;

    for (int matrix_size = 1; matrix_size < n; matrix_size += n / 10) {

        std::cout << std::endl << "n: " << matrix_size << std::endl;

        Matrix A = Matrix::random(matrix_size);

        for (int repetitions = 0; repetitions < max_reps; repetitions += rep_step) {
            for (int epsilon_ceros = 1; epsilon_ceros < max_epsilon_ceros ; epsilon_ceros++) {

                double epsilon = pow(1.0e-1, epsilon_ceros);

                Matrix x_0( random(A.rows(), 1));
                Matrix eigen_vector(A.rows(), 1);
                double eigen_value;

                auto begin = GET_TIME;
                tie(eigen_vector, eigen_value) =
                        power_method(x_0, A, epsilon); // calculate i_th eigen vector and it's value
                auto end = GET_TIME;
                auto calculation_time = GET_TIME_DELTA(begin, end);

                double result_mse = Matrix::mse(A * eigen_vector, eigen_vector * eigen_value);

                file << matrix_size << "," << epsilon << "," << repetitions<< "," << calculation_time << result_mse << std::endl;
                std::cout << "time: " << calculation_time << std::endl;

            }
        }
    }
}

