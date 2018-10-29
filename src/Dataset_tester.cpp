#include "Dataset_tester.h"

void Dataset_tester::checksumPCA(){
    vector<double> componentSums = vector<double>();
    //each column is a vector (a component so to speak)
    	for(int j = 0; j<myDataset.getPcaVecs().cols(); j++){
    		double component_sum = 0.0;
    		for(int i = 0; i<myDataset.getPcaVecs().rows() ; i++){
            component_sum+=fabs(myDataset.getPcaVecs()(i,j));
    	}
        componentSums.push_back(component_sum);
        cout << "done sum: "<< j <<" got "<<component_sum<<endl;
    }
    for(int k = 0; k < componentSums.size(); k++){
    	cout << k <<" sum: "<<componentSums[k]<<endl;
    }

    sort(componentSums.begin(), componentSums.end());

    for(int k = 0; k < componentSums.size(); k++){
    	cout << k <<" sum: "<<componentSums[k]<<endl;
    }
}

void Dataset_tester::write_down_two_components(){ 
    cout << "breaks here" << endl;
    myDataset.trainActualPCA(2);
    cout << myDataset.getTransformedTrainImages() << endl << endl;
    cout << myDataset.getTrainLabels() << endl;
}