#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <string>
#include <string.h>

#include "src/Dataset.h"

int main(int argc, char** argv){

    auto filter_out = [] (const int token, const FrecuencyVocabularyMap & vocabulary) {
            double token_frecuency = vocabulary.at(token);
            return token_frecuency < 0.01 || token_frecuency > 0.99;
        };
    VectorizedEntriesMap train_entries;
    VectorizedEntriesMap test_entries;
    build_vectorized_datasets(train_entries, test_entries, filter_out);
    int N = train_entries.begin()->second.bag_of_words.size();

    std::cerr
            << "N: " << N << std::endl
            << "Dataset de entrenamiento: " << train_entries.size() << " entradas" << std::endl
            << "Dataset de testeo: " << test_entries.size() << " entradas" << std::endl;

    return 0;

}