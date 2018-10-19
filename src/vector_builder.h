#ifndef VECTOR_BUILDER__H
#define VECTOR_BUILDER__H

#include <functional>
#include <iostream>

#include "types.h"
#include "reader.h"

#include "Matrix.h"

std::tuple<std::tuple<Matrix, Matrix>, std::tuple<Matrix, Matrix>> build_vectorized_datasets
        (const std::function<bool(int token, const FrecuencyVocabularyMap & vocabulary)> & filter_out);

#endif  // VECTOR_BUILDER__H

