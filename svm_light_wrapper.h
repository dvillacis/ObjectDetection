#ifndef __SVM_LIGHT_WRAPPER_LIB_H__
#define __SVM_LIGHT_WRAPPER_LIB_H__

#include <vector>
#include <string>
#include <fstream>

namespace SVMLight
{
    extern class SVMTrainer
    {
    private:
        std::fstream featuresFile_;
        std::string featuresFileName_;
    public:
        SVMTrainer(const std::string& featuresFileName);
        void writeFeatureVectorToFile(const std::vector<float>& featureVector, bool isPositive);
        void trainAndSaveModel(const std::string& modelFileName);
    };

    extern class SVMClassifier
    {
    public:
        SVMClassifier(const std::string& featuresFileName);
        std::vector<float> getDescriptorVector();
    };
}

#endif