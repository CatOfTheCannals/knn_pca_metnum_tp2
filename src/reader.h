#ifndef READER__H
#define READER__H

#include <fstream>
#include <iostream>
#include <sstream>
#include <string.h>
#include <string>
#include <stdexcept>

#include "types.h"

void read_entries(const std::string & entries_path, TokenizedEntriesMap & train_entries, TokenizedEntriesMap & test_entries);
FrecuencyVocabularyMap read_vocabulary();

#endif  // READER__H

