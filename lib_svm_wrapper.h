#ifndef __LIB_SVM_WRAPPER_LIB_H__
#define __LIB_SVM_WRAPPER_LIB_H__

#include <vector>
#include <string>
#include <fstream>

#include "svm.h"

using namespace std;

namespace LibSVM
{
    extern class SVMTrainer
    {
    private:
        fstream featuresFile_;
        string featuresFileName_;
        char* readline(FILE *input);
        void exit_input_error(int line_num);
    public:
        SVMTrainer(const string& featuresFileName);
        void writeFeatureVectorToFile(const vector<float>& featureVector, bool isPositive);
        void trainAndSaveModel(const string& modelFileName, const int& kernelType);
        const char* checkParameters(const struct svm_problem *prob, const struct svm_parameter *param);
        svm_model * trainModel(const struct svm_problem *prob, const struct svm_parameter *param);
    };

    extern class SVMClassifier
    {
    public:
        SVMClassifier(const string& featuresFileName);
        vector<float> getDescriptorVector();
    };
}

#endif