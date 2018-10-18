#ifndef VECTOR_BUILDER__H
#define VECTOR_BUILDER__H

#include <functional>
#include <iostream>

#include "types.h"
#include "reader.h"

void build_vectorized_datasets (
        VectorizedEntriesMap & train_vectorized_entries,
        VectorizedEntriesMap & test_vectorized_entries,
        const std::function<bool(int token, const FrecuencyVocabularyMap & vocabulary)> & filter_out);

#endif  // VECTOR_BUILDER__H

