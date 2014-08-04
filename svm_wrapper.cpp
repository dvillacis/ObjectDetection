#include <vector>
#include <ios>
#include <string>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include "svm_wrapper.h"

#define AUTO_TRAIN_SVM

using namespace std;
using namespace cv;

static vector<string> split(const string& s, const string& delim, const bool keep_empty = true) {
  vector<string> result;
  if (delim.empty()) {
    result.push_back(s);
    return result;
  }
  string::const_iterator substart = s.begin(), subend;
  while (true) {
    subend = search(substart, s.end(), delim.begin(), delim.end());
    string temp(substart, subend);
    if (keep_empty || !temp.empty()) {
      result.push_back(temp);
    }
    if (subend == s.end()) {
      break;
    }
    substart = subend + delim.size();
  }
  return result;
}

namespace LibSVM 
{
    class SVMImpl {
    private:
        MySVM* svm;
        CvSVMParams* params;
        SVMImpl() {
            svm = new MySVM;
            params = new CvSVMParams(
                CvSVM::C_SVC,   // Type of SVM; using N classes here
                CvSVM::LINEAR,  // Kernel type
                3,            // Param (degree) for poly kernel only
                1.0,            // Param (gamma) for poly/rbf kernel only
                1.0,            // Param (coef0) for poly/sigmoid kernel only
                10,             // SVM optimization param C
                0,              // SVM optimization param nu (not used for N class SVM)
                0,              // SVM optimization param p (not used for N class SVM)
                NULL,           // class weights (or priors)
                cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 0.000001)
            );
            
        }

        virtual ~SVMImpl() {
            // Cleanup area

        }

    public:

        Mat trainingData, trainingClass;
        Mat varIdx, sampleIdx;

        static SVMImpl* getInstance() {
            static SVMImpl theInstance;
            return &theInstance;
        }

        inline void saveModelToFile(const string& modelFileName) {
            svm->save(modelFileName.c_str());
        }

        void loadModelFromFile(const string& modelFileName) {
            svm->load(modelFileName.c_str());
        }

        // read in a problem (in SVMLight format)
        void read_problem(const string& filename) {
            LOG(INFO) << "Loading features into memory";
            vector<float> labels;
            vector<vector<float> > data;
            ifstream featureFile;
            featureFile.open(filename);
            string line;
            if(featureFile.is_open()){
                while(getline(featureFile,line)){
                    vector<string> parts = split(line," ");
                    vector<float> features;
                    labels.push_back(atof(parts.at(0).c_str()));
                    for(int i = 1; i < parts.size(); ++i){
                        features.push_back(atof(split(parts[i],":").at(1).c_str()));
                    }
                    data.push_back(features);
                }
                featureFile.close();
            }

            //Creating the Mat for training
            trainingData = Mat(data.size(), data.at(0).size(), CV_32FC1);
            trainingClass = Mat(labels.size(), 1, CV_32FC1);
            for(int i = 0; i < data.size(); ++i){
                trainingClass.at<float>(i,0) = labels.at(i);
                for(int j = 0; j < data.at(i).size(); ++j){
                    trainingData.at<float>(i,j) = data.at(i).at(j);
                }
            }
            
        }

        void trainModel(const int& kernelType){
            params->kernel_type = kernelType;
        #ifdef AUTO_TRAIN_SVM
            LOG(INFO) << "Staring training " << trainingData.rows << " examples with " << trainingData.cols << " features";
            LOG(INFO) << "Finding optimal parameters to use";
            svm->train_auto(trainingData,trainingClass,varIdx,sampleIdx,*params,10);
            LOG(INFO) << "The optimal parameters are: degree: " << params->degree << ", gamma: " <<
                params->gamma << ", coef0: " << params->coef0 << ", C: " << params->C << ", nu: " <<
                params->nu << ", p: " << params->p;
        #else
            LOG(INFO) << "Training using default parameters";
            svm->train(trainingData,trainingClass,varIdx,sampleIdx,*params);
        #endif
            LOG(INFO) << "Training completed";
            LOG(INFO) << "Number of support vectors found: " << svm->get_support_vector_count();
        }

        void getSingleDetectingVector(vector<float>& singleDetectorVector) {
            int sv_count = svm->get_support_vector_count();
            const CvSVMDecisionFunc* df = svm->get_decision_function();
            const double* alphas = df[0].alpha;
            double rho = df[0].rho;
            int var_count = svm->get_var_count();
            singleDetectorVector.resize(var_count,0);
            for(unsigned int r = 0; r < (unsigned)sv_count; r++){
                float my_alpha = alphas[r];
                const float* v = svm->get_support_vector(r);
                for(int j = 0; j < var_count; j++,v++){
                    singleDetectorVector[j] += (-my_alpha)*(*v);
                }
            }
            singleDetectorVector.push_back(rho);
        }

    };

    // SVMTrainer & SVMClassifier implementations:

    SVMTrainer::SVMTrainer(const string& featuresFileName)
    {
        // use the C locale while creating the model file:
        setlocale(LC_ALL, "C");

        featuresFileName_ = featuresFileName;
        featuresFile_.open(featuresFileName_.c_str(), ios::out);
    }

    void SVMTrainer::writeFeatureVectorToFile(const vector<float>& featureVector, bool isPositive)
    {
        featuresFile_ << ((isPositive) ? "+1" : "-1");
        for (unsigned int feature = 0; feature < featureVector.size(); ++feature) {
            featuresFile_ << " " << (feature + 1) << ":" << featureVector.at(feature);
        }
        featuresFile_ << endl;
    }

    void SVMTrainer::trainAndSaveModel(const string& modelFileName, const int& kernelType)
    {
        if (featuresFile_.is_open())
            featuresFile_.close();

        SVMImpl::getInstance()->read_problem(featuresFileName_);
        LOG(INFO) << "Problem read successfully";
        SVMImpl::getInstance()->trainModel(kernelType);
        LOG(INFO) << "Model trained";
        SVMImpl::getInstance()->saveModelToFile(modelFileName);
    }

    SVMClassifier::SVMClassifier(const string& modelFilename)
    {
         SVMImpl::getInstance()->loadModelFromFile(modelFilename);
    }
        
    vector<float> SVMClassifier::getDescriptorVector()
    {
        vector<float> descriptorVector;   
        SVMImpl::getInstance()->getSingleDetectingVector(descriptorVector);
        return descriptorVector;
    }
}