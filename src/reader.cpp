#include "reader.h"

void read_entries(TokenizedEntriesMap & train_entries, TokenizedEntriesMap & test_entries) {
    /**
     *  Parsea el archivo de reviews tokenizadas
     *  El archivo en cuestión no debe tener una línea vacía al final
     **/
    std::cerr << "Levantando dataset..." << '\r';
    std::string line;
    std::ifstream infile;
    infile.open ("../datos/imdb_tokenized.txt");
    if (infile.fail()) throw std::runtime_error("Ocurrió un error al abrir el archivo.");

    while (std::getline(infile,line)) {
        // Leo una línea y cargo una entrada
        int review_id(stoi(std::string(strtok(&line[0u], ","))));
        std::string dataset(strtok(NULL, ","));
        std::string polarity(strtok(NULL, ","));

        TokenizedEntry entry;
        char * pch = strtok(NULL, ",");
        while (pch != NULL) {
            entry.tokens.push_back(stoi(std::string(pch)));
            pch = strtok(NULL, ",");
        }

        entry.is_positive = polarity == "pos";
        if (dataset == "test") test_entries[review_id] = entry;
        else train_entries[review_id] = entry;
    }

    infile.close();

    std::cerr << "                     " << '\r';
}

FrecuencyVocabularyMap read_vocabulary() {
    /**
     *  Parsea el archivo de vocabulario
     *  El archivo en cuestión no debe tener una línea vacía al final
     **/
    std::cerr << "Levantando vocabulario" << '\r';
    FrecuencyVocabularyMap vocabulary;
    std::string line;
    std::ifstream infile;
    infile.open ("../datos/vocab.csv");
    if (infile.fail()) throw std::runtime_error("Ocurrió un error al abrir el archivo.");

    std::getline(infile,line);  // Sacamos el header del csv
    while (std::getline(infile,line)) {  // Leemos una línea
        // Leo las últimas dos columnas
        std::string penultime, last;
        char * pch = strtok(&line[0u]," ,");
        while (pch != NULL) { penultime = last; last = pch; pch = strtok (NULL, " ,"); }

        // Cargo la entrada
        int token_id(stoi(last));
        double token_frecuency(stof(penultime));
        vocabulary[token_id] = token_frecuency;
    }

    infile.close();

    std::cerr << "                      " << '\r';
    return vocabulary;
}