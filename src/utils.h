#ifndef TP2_METODOS_UTILS_H
#define TP2_METODOS_UTILS_H

#include <iostream>
#include <iostream>
#include <fstream>
#include <sstream>
#include "chrono"
#include <stdlib.h>
#include <tuple>

#define GET_TIME std::chrono::high_resolution_clock::now()
#define GET_TIME_DELTA(begin, end) \
     std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()


#endif //TP2_METODOS_UTILS_H
