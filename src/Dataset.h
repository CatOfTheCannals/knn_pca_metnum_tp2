#ifndef TP2_METODOS_DATASET_H
#define TP2_METODOS_DATASET_H

#include <fstream>
#include <string>
#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <time.h>

#include "Matrix.h"
#include "Pca.h"
#include "kNN.h"
#include "Metrics.h"

#include "ppmloader.h"

class Dataset {
public:

    Dataset() : _images(), _targets(){}

    Dataset(const Matrix& _images, const Matrix& _targets) : _images(_images), _targets(_targets){}

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
        _targets = Matrix(person_ids.size(), 1);

        // initialize feature and taget matrices
        for(int i = 0; i < _images.rows(); i ++){
            _targets.setIndex(i, 0, person_ids[i]);

            // this shit below is done just to initialize a matrix row
            width = 0;
            height = 0;
            filename = filePath + imagePaths[i];
            pt = PPM_LOADER_PIXEL_TYPE_INVALID;
            bool ret = LoadPPMFile(&data, &width, &height, &pt, filename.c_str());
            assert(ret || width != 0|| height != 0);

            for (int h = 0; h < height; ++h){
                for (int w = 0; w < width; ++w){
                    int colIndex = h * width + w;
                    _images.setIndex(i, colIndex, (unsigned int)data[colIndex]);
                }
            }
        }
    };

    Matrix getImages() const;
    Matrix getTargets() const;
    Matrix getPcaVecs() const;
    Matrix getPcaLambdas() const;

    void shuffle();
    void trainPca(int alpha, double epsilon);
    Matrix kNN_predict(int k) const;
    void splitTrainFromTest(double testPercentage);

private:
    Matrix _images;
    Matrix _targets;
    Matrix _pcaVecs;
    Matrix _pcaLambdas;
    Matrix _testImages;
    Matrix _testLabels;
    Matrix _trainImages;
    Matrix _trainLabels;
};


#endif //TP2_METODOS_DATASET_H
