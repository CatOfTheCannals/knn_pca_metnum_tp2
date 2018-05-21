#ifndef TP2_METODOS_DATASET_H
#define TP2_METODOS_DATASET_H

#include "Matrix.h"
#include "ppmloader.h"
#include <fstream>
#include <string>
#include <sstream>
#include <iostream>

class Dataset {
public:
    Dataset() : _images() {}

    Dataset(string filePath, string fileName) {
        ifstream f_test(filePath + fileName);
        assert(f_test.is_open());
        string line, imagePath, personNumber;
        std::vector<string> imagePaths;
        std::vector<int> person_ids;
        // parse whole file
        while( getline(f_test, line) ){
            istringstream lineStream(line);
            lineStream >> imagePath >> personNumber;
            imagePath.pop_back();
            personNumber.pop_back();
            imagePaths.push_back(imagePath);
            person_ids.push_back(stoi(personNumber));
        }

        // use last parsed so as to know the amount of samples and image size
        uchar* data = NULL;
        int width = 0, height = 0;
        string filename(filePath + imagePath);
        PPM_LOADER_PIXEL_TYPE pt = PPM_LOADER_PIXEL_TYPE_INVALID;
        bool ret = LoadPPMFile(&data, &width, &height, &pt, filename.c_str());
        assert(ret || width != 0|| height != 0);


        _images =  Matrix(imagePaths.size(), width * height);
        _target = Matrix(person_ids.size(), 1);

        // initialize feature and taget matrices
        for(int i = 0; i < _images.rows(); i ++){
            _target.setIndex(i, 0, person_ids[i]);

            // this shit below is done just to initialize a matrix row
            width = 0;
            height = 0;
            filename = filePath + imagePaths[i];
            pt = PPM_LOADER_PIXEL_TYPE_INVALID;
            bool ret = LoadPPMFile(&data, &width, &height, &pt, filename.c_str());
            assert(ret || width != 0|| height != 0);

            for (int h = 0; h < height; ++h){
                for (int w = 0; w < width; ++w){
                    // %90 shure that get_pixel_average is not necessary
                    // std::cout << get_pixel_average(data, h, w, height, width) << " ";
                    int colIndex = h * width + w;
                    _images.setIndex(i, colIndex, (unsigned int)data[colIndex]);
                }
            }
        }
    };

    Matrix getImages() const;

private:
    Matrix _images;
    Matrix _target;
};


#endif //TP2_METODOS_DATASET_H
