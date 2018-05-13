#ifndef __MATRIX_HPP__
#define __MATRIX_HPP__

#include <iostream>
#include <vector>
#include <string>
#include <assert.h>
#include <tuple>
#include <math.h>

using namespace std;

class Matrix {
public:
    Matrix() : _rows(0), _cols(0) {}

    // Constructor of Matrix Class initallized with 0 on every position.
    Matrix(int rows, int columns) : _rows(rows), _cols(columns){ //fixme: no se para que hago lo que hago dps de los dos puntos, creo que con las sgtes dos líneas,  no es necesario.
        _rows = rows;
        _cols = columns;
        _matrix = new double[_rows*_cols]();  // dynamically allocates memory using new
        for (int i= 0; i < _rows*_cols; ++i) {
            _matrix[i] = 0;
        }
    }

    Matrix(const Matrix & other) : _rows(other._rows), _cols(other._cols)
    {
        const int arr_size = _rows * _cols;
        _matrix = new double[arr_size];
        std::copy(other._matrix, other._matrix + arr_size, _matrix);
    }

    int rows() const;
    int cols() const;
    int size() const;
    double operator()(std::size_t row_idx, std::size_t col_idx) const;
    double operator()(std::size_t idx) const;
    void setIndex(int i, int j, double value); //fixme: completaro
    Matrix operator+(const Matrix& matrix) const;
    Matrix operator-(const Matrix& matrix) const;
    Matrix operator*(const double& scalar) const; //scalar multiplication
    Matrix operator/(const double& scalar) const; //scalar division
    Matrix transpose() ;
    void swapRows(int i1, int i2);
    Matrix getRow(int index);
    std::tuple<int, int> shape() const;
    bool operator==(const Matrix& other) const;
    Matrix subMatrix(int i1, int i2, int j1, int j2) const;
    Matrix multiply(const Matrix b);
    static Matrix identity(int n);
    Matrix abs();
    std::tuple<int, int> maxCoeff();
    friend std::ostream& operator<<(std::ostream& o, const Matrix& a);
    bool isApproximate(const Matrix b, double epsilon) const;
    double squared_norm() const;
    double norm() const;

private:
    int _rows;
    int _cols;
    double *_matrix; //pointer of type int to the location of the matrix.
};

#endif //__DYN_MATRIX_HPP__}

