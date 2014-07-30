#ifndef __SVM_LIGHT_WRAPPER_LIB_H__
#define __SVM_LIGHT_WRAPPER_LIB_H__

#include <vector>
#include <string>
#include <fstream>

using namespace std;

namespace SVMLight
{
    extern class SVMTrainer
    {
    private:
        fstream featuresFile_;
        string featuresFileName_;
    public:
        SVMTrainer(const string& featuresFileName);
        void writeFeatureVectorToFile(const vector<float>& featureVector, bool isPositive);
        void trainAndSaveModel(const string& modelFileName, const int& kernelType);
    };

    extern class SVMClassifier
    {
    public:
        SVMClassifier(const string& featuresFileName);
        vector<float> getDescriptorVector();
    };
}

#endif