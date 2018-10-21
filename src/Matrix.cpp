#include "Matrix.h"

int Matrix::rows() const{
    return _rows;
}

int Matrix::cols() const {
    return _cols;
}

int Matrix::size() const {
    return _rows * _cols;
}


double Matrix::operator()(std::size_t row_idx, std::size_t col_idx) const
{
    assert(row_idx < this->_rows && col_idx < this->_cols);
    std::size_t idx = row_idx * this->_cols + col_idx;
    return this->_matrix[idx];
}

double Matrix::operator()(std::size_t idx) const
{
    assert(this->_rows == 1 || this->_cols == 1);
    assert(idx < std::max(this->_rows, this->_cols));
    return this->_matrix[idx];
}

void Matrix::operator=(const Matrix& other) {
    _rows = other._rows;
    _cols = other._cols;
    if(_matrix != nullptr) delete[] _matrix;
    const int arr_size = _rows * _cols;
    _matrix = new double[arr_size];
    std::copy(other._matrix, other._matrix + arr_size, _matrix);
}

void Matrix::setIndex(int i, int j, double value){
    assert(0 <= i && i < _rows && 0 <= j && j < _cols);
    int position = _cols * i + j;
    _matrix[position] = value;

}

Matrix Matrix::operator+(const Matrix& matrix) const{
    assert(this->_cols == matrix.cols() && this->_rows == matrix.rows());
    Matrix res = (*this);

    for(int i=0; i < res.rows(); ++i){
        for(int j=0; j < res.cols(); ++j){
            res.setIndex(i, j, res(i, j) + matrix(i, j));
        }
    }
    return res;
}

Matrix Matrix::operator-(const Matrix& matrix) const{
    assert(this->_cols == matrix.cols() && this->_rows == matrix.rows());
    Matrix res = (*this);

    for(int i=0; i < res.rows(); ++i){
        for(int j=0; j < res.cols(); ++j){
            res.setIndex(i, j, res(i, j) - matrix(i, j));
        }
    }
    return res;
}

Matrix Matrix::operator*(const double& scalar) const{
    Matrix res = (*this);
    for(int i=0; i < res.rows(); ++i){
        for(int j=0; j < res.cols(); ++j){
            res.setIndex(i, j, res(i, j) * scalar);
        }
    }
    return res;
}

Matrix Matrix::operator*(const Matrix& b) const{
    assert(this->_cols == b.rows());
    Matrix result = Matrix(this->_rows, b.cols());
    for (int i = 0; i < this->rows(); ++i) {
        for (int j = 0; j < b.cols(); ++j) {
            double temp = 0;
            for (int k = 0; k < this->_cols; ++k) {
                temp = temp + (*this)(i, k) * b(k, j);
            }
            result.setIndex(i, j, temp);
            temp = 0;
        }

    }
    return result;
}

void Matrix::show_matrix(){
    cout << endl;
    for(int i = 0 ; i < _cols ; i++){
        for(int j = 0 ; j < _rows ; j++){
            cout << (*this)(i,j) << " ";
        }
        cout << endl;
    }
}

Matrix Matrix::mt_times_m() const {
    //X.transpose()*X

    Matrix result = Matrix(this->_cols, this->_cols);
    int index = 0;
    for (int i = 0; i < this->_cols; i++) {
        double* i_column = new double[this->_rows];
        for (int k = 0; k < this->_rows; k++) {
            i_column[k] = (*this)(k, i);
        }

        for (int j = 0; j < this->_cols; j++) {

            double temp = 0;
            //actual multiplication
            for (int k = 0; k < this->_rows; k++) {
                temp += i_column[k] * (*this)(k, j);
            }

            result._matrix[index] = temp;
            temp = 0;
            index++;
        }
        delete i_column;
    }
    return result;
}

Matrix Matrix::mt_times_m_cache() const {
    //X.transpose()*X

    Matrix result = Matrix(this->_cols, this->_cols);
    for (int i = 0; i < this->_cols; i++) {
        double* i_column = new double[this->_rows];

        for (int k = 0; k < this->_rows; k++) {
            i_column[k] = (*this)(k, i);
        }

        for (int j = 0; j < this->_cols; j++) {
            double temp = 0;
            int is_new = 0;
            // more cache magic
            double* j_column = nullptr;
            if(i==j){
                j_column = i_column;
            } else {
                is_new = 1;
                j_column = new double[this->_rows];
                for (int k = 0; k < this->_rows; k++) {
                    j_column[k] = (*this)(k, j);
                }
            }
            //actual multiplication
            for (int k = 0; k < this->_rows; k++) {
                temp += i_column[k] * j_column[k];
            }
            int index = result._cols * i + j;
            result._matrix[index] = temp;
            temp = 0;
            if(is_new==1){delete j_column;}
        }
        delete i_column;
    }
    return result;
}

Matrix Matrix::operator/(const double& scalar) const{
    Matrix res = (*this);
    for(int i=0; i < res.rows(); ++i){
        for(int j=0; j < res.cols(); ++j){
            res.setIndex(i, j, res(i, j) / scalar);
        }
    }
    return res;
}

Matrix Matrix::transpose() const{
    Matrix res(this->cols(), this->rows());
    for(int i = 0; i < this->rows() ; i++){
        for(int j = 0; j < this->cols() ; j++){
            res.setIndex(j,i, (*this)(i,j));
        }
    }
    return res;
}

void Matrix::swapRows(int i1, int i2) {
    assert(i1 < _rows && i2 < _rows);
    if(i1 != i2){
        double  *tempRow = new double[_cols];
        double *start_index_1 = _matrix + i1 * _cols;
        double *end_index_1 = start_index_1 + _cols;
        double *start_index_2 = _matrix + i2 * _cols;
        double *end_index_2 = start_index_2 + _cols;

        std::copy(start_index_1, end_index_1, tempRow);
        std::copy(start_index_2, end_index_2, start_index_1);
        std::copy(tempRow, tempRow + _cols, start_index_2);
    }
}

Matrix Matrix::getRow(int index) const{
    Matrix row(1, _cols);
    for (int i= 0; i < _cols; ++i) {
        row._matrix[i] = _matrix[index * _cols + i];
    }
    return row;
}

void Matrix::setRow(int index, const Matrix& row) {
    assert(row.rows() ==1 && row.cols() < _cols);
    for (int j= 0; j < _cols; ++j) {
        (*this).setIndex(index, j, row(j));
    }
}

std::tuple<int, int> Matrix::shape() const {
    return std::make_tuple(this->_rows, this->_cols);
}

bool Matrix::operator==(const Matrix& other) const{
    if (this->_rows != other.rows() || this->_cols != other.cols()) {
        return false;
    }
    else {
        for(int i = 0; i < _rows * _cols; i++){
            if(this->_matrix[i] != other._matrix[i]){
                return false;
            }
        }
    }
    return true;
}

Matrix Matrix::subMatrix(int i1, int i2, int j1, int j2 ) const{
    /*
     * returns the matrix between (i1, i2) rows and (j1, j2) cols
     * indexes i2 and j2 will be part of the answer
     */

    assert(i1 <= i2 && j1 <= j2);
    assert(-1 < i1 && -1 < j2);
    assert(i2 < _rows && j2 < _cols);

    int res_rows = i2 - i1 + 1;
    int res_cols = j2 - j1 + 1;

    int index;

    Matrix res(res_rows, res_cols);
    for(int i = 0; i < res_rows ; i++){
        for(int j = 0; j < res_cols ; j++){
            index = (i + i1) * _cols + j1 + j;
            res._matrix[i * res_cols + j] = _matrix[index];
        }
    }

    return res;
}


Matrix Matrix::identity(int n){
    assert(0 < n);
    Matrix res(n, n);
    for(int i = 0; i < n; i++){
        res.setIndex(i, i, 1);
    }
    return res;
}

Matrix Matrix::random(int n){
    return Matrix::random(n, n);
}
Matrix Matrix::random(int height, int width){
    Matrix random_matrix(height, width);
    for(int i = 0; i < random_matrix.rows(); i++){
        for(int j = 0; j < random_matrix.cols(); j++){
            random_matrix.setIndex(i, j, rand() % 255);
        }
    }
    return random_matrix;
}



std::tuple<int, int> Matrix::maxCoeff() {
    int res_x, res_y;
    double max = (*this)(0,0);
    for (int i = 0; i < this->rows(); ++i) {
        for (int j = 0; j < this->cols(); ++j) {
            if((*this)(i,j) > max){
                max = (*this)(i,j);
                res_x = i;
                res_y = j;
            }
        }
    }
    return std::make_tuple(res_x, res_y);
};

Matrix Matrix::abs() {
    Matrix res(this->rows(), this->cols());

    for (int i = 0; i < this->rows(); ++i) {
        for (int j = 0; j < this->cols(); ++j) {
            double val = (*this)(i,j);
            if((*this)(i,j) < 0){
                res.setIndex(i, j, val * -1);
            } else {
                res.setIndex(i, j, val);
            }
        }
    }

    return res;
}

double Matrix::mse(const Matrix& v1, const Matrix& v2) {
    assert(v1.rows() == 1 || v1.cols() == 1);
    assert(v2.rows() == 1 || v2.cols() == 1);
    assert(v1.rows() == v2.rows()|| v1.cols() == v2.cols());

    int n = std::max(v1.rows(), v1.rows());
    double sum = 0;
    for(int i = 1; i < n; i++) {
        sum += pow(v1(i) - v2(i), 2);
    }
    return sum / n;
}

std::ostream& operator<<(std::ostream& o, const Matrix& a)
{
    for (std::size_t i = 0; i < a.rows(); i++) {
        for (std::size_t j = 0; j < a.cols(); j++) {
            o << a(i, j) << ' ';
        }
        if (!(i / a.rows())) {
            o << endl;
        }
    }
    return o;

}

bool Matrix::isApproximate(const Matrix b, double epsilon) const{
    Matrix a_copy = (*this);
    Matrix b_copy = b;
    auto diff = (a_copy + b_copy * (-1)).abs();

    for (int i = 0; i < b_copy.rows(); i++) {
        for (int j = 0; j < b_copy.cols(); j++) {
            if(diff(i,j) > epsilon){
                return false;
            }
        }
    }
    return true;
}

double Matrix::squared_norm() const {
    assert(this->_rows == 1 || this->_cols == 1);
    double res = 0;
    auto size = this->size();
    for (std::size_t i = 0; i < size; i++) {
        res += (double)((*this)(i) * (*this)(i));
    }

    return res;
}

double Matrix::norm() const {
    assert(this->_rows == 1 || this->_cols == 1);
    return sqrt(this->squared_norm());
}

Matrix Matrix::vecOfRowsToMatrix(const std::vector<Matrix> vecOfRows) {
    assert(vecOfRows.size() > 0);
    Matrix hstack(vecOfRows.size(), vecOfRows[0].cols());
    for(int i = 0; i < hstack.rows(); i++) {
        for(int j = 0; j < hstack.cols(); j++) {
            hstack.setIndex(i, j, vecOfRows[i](j));
        }
    }

    return hstack;
}
