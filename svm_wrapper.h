#ifndef __LIB_SVM_WRAPPER_LIB_H__
#define __LIB_SVM_WRAPPER_LIB_H__

#include <vector>
#include <string>
#include <fstream>

#include <opencv2/ml/ml.hpp>

using namespace std;

namespace LibSVM
{
    class MySVM : public CvSVM{
    public:
        CvSVMDecisionFunc* get_decision_function(){
            return decision_func;
        }

    };

    class SVMTrainer
    {
    private:
        fstream featuresFile_;
        string featuresFileName_;
    public:
        SVMTrainer(const string& featuresFileName);
        void writeFeatureVectorToFile(const vector<float>& featureVector, bool isPositive);
        void trainAndSaveModel(const string& modelFileName, const CvSVMParams* myParams);
        void closeFeaturesFile();
    };

    class SVMClassifier
    {
    public:
        SVMClassifier(const string& featuresFileName);
        vector<float> getDescriptorVector();
    };
}

#endif