#include <vector>
#include <ios>
#include <string>
#include <fstream>
#include <iostream>
#include <stdio.h>

#include "lib_svm_wrapper.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))


using namespace std;

namespace LibSVM 
{
    extern "C" 
    {
        #include "svm.h" 
    }

    static char *line = NULL;
    static int max_line_len;
    
    // more or less original wrapper of LibSVM taken from https://github.com/DaHoC/trainHOG
    class LibSVMImpl {
    private:
        LibSVMImpl() {
            // default values
            param.svm_type = C_SVC;
            param.kernel_type = RBF;
            param.degree = 3;
            param.gamma = 0;    // 1/num_features
            param.coef0 = 0;
            param.nu = 0.5;
            param.cache_size = 100;
            param.C = 1;
            param.eps = 1e-3;
            param.p = 0.1;
            param.shrinking = 1;
            param.probability = 0;
            param.nr_weight = 0;
            param.weight_label = NULL;
            param.weight = NULL;
            cross_validation = 0;
        }

        virtual ~LibSVMImpl() {
            // Cleanup area
            // Free the memory used for the cache
            svm_free_and_destroy_model(&model);
            svm_destroy_param(&param);
            free(prob.y);
            free(prob.x);
            free(x_space);
            free(line);
        }

        static char* readline(FILE *input)
        {
            int len;
            
            if(fgets(line,max_line_len,input) == NULL)
                return NULL;

            while(strrchr(line,'\n') == NULL)
            {
                max_line_len *= 2;
                line = (char *) realloc(line,max_line_len);
                len = (int) strlen(line);
                if(fgets(line+len,max_line_len-len,input) == NULL)
                    break;
            }
            return line;
        }

        void exit_input_error(int line_num)
        {
            fprintf(stderr,"Wrong input format at line %d\n", line_num);
            exit(1);
        }

    public:

        svm_parameter param;     // set by parse_command_line
        svm_problem prob;        // set by read_problem
        struct svm_model *model;
        svm_node *x_space;
        int cross_validation;
        int nr_fold;

        static LibSVMImpl* getInstance() {
            static LibSVMImpl theInstance;
            return &theInstance;
        }

        inline bool saveModelToFile(const string& _modelFileName, const svm_model* _model) {
            cout << "Trying to save the model" << endl;
            return svm_save_model(_modelFileName.c_str(), _model);
        }

        void loadModelFromFile(const string& _modelFileName) {
            this->model = svm_load_model(const_cast<char*>(_modelFileName.c_str()));
        }

        // read in a problem (in SVMLight format)
        void read_problem(const string& filename) {
            // Reads and parses the specified file
            int elements, max_index, inst_max_index, i, j;
            FILE *fp = fopen(filename.c_str(),"r");
            char *endptr;
            char *idx, *val, *label;

            if(fp == NULL)
            {
                fprintf(stderr,"can't open input file %s\n",filename.c_str());
                exit(1);
            }

            prob.l = 0;
            elements = 0;

            max_line_len = 1024;
            line = Malloc(char,max_line_len);
            while(readline(fp)!=NULL)
            {
                char *p = strtok(line," \t"); // label

                // features
                while(1)
                {
                    p = strtok(NULL," \t");
                    if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
                        break;
                    ++elements;
                }
                ++elements;
                ++prob.l;
            }
            rewind(fp);

            prob.y = Malloc(double,prob.l);
            prob.x = Malloc(struct svm_node *,prob.l);
            x_space = Malloc(struct svm_node,elements);

            max_index = 0;
            j=0;
            for(i=0;i<prob.l;i++)
            {
                inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
                readline(fp);
                prob.x[i] = &x_space[j];
                label = strtok(line," \t\n");
                if(label == NULL) // empty line
                    exit_input_error(i+1);

                prob.y[i] = strtod(label,&endptr);
                if(endptr == label || *endptr != '\0')
                    exit_input_error(i+1);

                while(1)
                {
                    idx = strtok(NULL,":");
                    val = strtok(NULL," \t");

                    if(val == NULL)
                        break;

                    errno = 0;
                    x_space[j].index = (int) strtol(idx,&endptr,10);
                    if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
                        exit_input_error(i+1);
                    else
                        inst_max_index = x_space[j].index;

                    errno = 0;
                    x_space[j].value = strtod(val,&endptr);
                    if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
                        exit_input_error(i+1);

                    ++j;
                }

                if(inst_max_index > max_index)
                    max_index = inst_max_index;
                x_space[j++].index = -1;
            }

            if(param.gamma == 0 && max_index > 0)
                param.gamma = 1.0/max_index;

            if(param.kernel_type == PRECOMPUTED)
                for(i=0;i<prob.l;i++)
                {
                    if (prob.x[i][0].index != 0)
                    {
                        fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
                        exit(1);
                    }
                    if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
                    {
                        fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
                        exit(1);
                    }
                }

            fclose(fp);
        }

        const char* checkParameters(const struct svm_problem *prob, const struct svm_parameter *param){
            return svm_check_parameter(prob,param);
        }

        svm_model * trainModel(const struct svm_problem *prob, const struct svm_parameter *param){
            return svm_train(prob,param);
        }

        /**
         * Generates a single detecting feature vector (v1 | b) from the trained support vectors, for use e.g. with the HOG algorithm
         * vec1 = sum_1_n (alpha_y*x_i). (vec1 is a 1 x n column vector. n = feature vector length )
         * @param singleDetectorVector resulting single detector vector for use in openCV HOG
         * @param singleDetectorVectorIndices
         */
        void getSingleDetectingVector(vector<float>& singleDetectorVector, vector<unsigned int>& singleDetectorVectorIndices) {
            // Now we use the trained svm to retrieve the single detector vector
                printf("Calculating single detecting feature vector out of support vectors (may take some time)\n");
                singleDetectorVector.clear();
                singleDetectorVectorIndices.clear();
                printf("Total number of support vectors: %d \n", model->l);
                //        printf("Number of SVs for each class: %d \n", _model->nr_class);
                double b = -(model->rho[0]); // This is the b value from the SVM, assumes that first the positive labels are read in (otherwise, use double b = (*_model->rho); )
                printf("b: %+3.5f\n", b);
                // Walk over every support vector and build a single vector
                for (unsigned long ssv = 0; ssv < model->l; ++ssv) { // Walks over available classes (e.g. +1, -1 representing positive and negative training samples)
                    //printf("Support vector #%lu \n", ssv);
                    // Retrive the current support vector from the training set
                    svm_node* singleSupportVector = model->SV[ssv]; // Get next support vector ssv==class, 2nd index is the component of the SV
                    //            _prob->x[singleSupportVector->index];
                    // sv_coef[i] = alpha[i]*sign(label[i]) = alpha[i] * y[i], where i is the training instance, y[i] in [+1,-1]
                    double alpha = model->sv_coef[0][ssv];
                    int singleVectorComponent = 0;
                    while (singleSupportVector[singleVectorComponent].index != -1) { // index=UINT_MAX indicates the end of the array
                        //    if (singleVectorComponent > 3777)
                        //        printf("\n-->%d", singleVectorComponent);
                        //                printf("Support Vector index: %u, %+3.5f \n", singleSupportVector[singleVectorComponent].index, singleSupportVector[singleVectorComponent].value);
                        if (ssv == 0) { // During first loop run determine the length of the support vectors and adjust the required vector size
                            singleDetectorVector.push_back(singleSupportVector[singleVectorComponent].value * alpha);
                            //                    printf("-%d", singleVectorComponent);
                            singleDetectorVectorIndices.push_back(singleSupportVector[singleVectorComponent].index); // Holds the indices for the corresponding values in singleDetectorVector, mapping from singleVectorComponent to singleSupportVector[singleVectorComponent].index!
                        } else {
                            if (singleVectorComponent > singleDetectorVector.size()) { // Catch oversized vectors (maybe from differently sized images?)
                                printf("Warning: Component %d out of range, should have the same size as other/first vector\n", singleVectorComponent);
                            } else
                                singleDetectorVector.at(singleVectorComponent) += (singleSupportVector[singleVectorComponent].value * alpha);
                        }
                        singleVectorComponent++;
                    }
        }

        // This is a threshold value which is also recorded in the lear code in lib/windetect.cpp at line 1297 as linearbias and in the original paper as constant epsilon, but no comment on how it is generated
        singleDetectorVector.push_back(b); // Add threshold
        singleDetectorVectorIndices.push_back(-1); // Add maximum unsigned int as index indicating the end of the vectortectorVector.at(model->totwords) = -model->b; /** @NOTE the minus sign! */
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
        const char* errorMsg;

        //Specify the kernel we want to use for the classification task
        LibSVMImpl::getInstance()->param.kernel_type = kernelType;

        if (featuresFile_.is_open())
            featuresFile_.close();

        LibSVMImpl::getInstance()->read_problem(featuresFileName_);
        cout << "Problem read successfully ... " << endl;

        errorMsg = LibSVMImpl::getInstance()->checkParameters(&LibSVMImpl::getInstance()->prob,&LibSVMImpl::getInstance()->param);
        if(errorMsg)
        {
            fprintf(stderr,"ERROR: %s\n",errorMsg);
            exit(1);
        }
        cout << "Parameters checked correctly ..." << endl;

        svm_model* model = LibSVMImpl::getInstance()->trainModel(&LibSVMImpl::getInstance()->prob,&LibSVMImpl::getInstance()->param);

        cout << "Model trained" << endl;

        if(LibSVMImpl::getInstance()->saveModelToFile(modelFileName,model)){
            fprintf(stderr, "can't save model to file %s\n", modelFileName.c_str());
            exit(1);
        }

        cout << "Model saved" << endl;
    }

    SVMClassifier::SVMClassifier(const string& modelFilename)
    {
         LibSVMImpl::getInstance()->loadModelFromFile(modelFilename);
    }
        
    vector<float> SVMClassifier::getDescriptorVector()
    {
        vector<float> descriptorVector;
        vector<unsigned int> descriptorVectorIndices;     
        LibSVMImpl::getInstance()->getSingleDetectingVector(descriptorVector, descriptorVectorIndices);
        return descriptorVector;
    }
}