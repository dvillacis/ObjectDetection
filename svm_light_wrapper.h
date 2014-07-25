#ifndef SVM_LIGHT_WRAPPER_H
#define SVM_LIGHT_WRAPPER_H

#include <stdio.h>
#include <vector>

namespace svm_light_wrapper{
  extern "C"{
    #include "svm_common.h"
    #include "svm_learn.h"
  }
}

using namespace svm_light_wrapper;

class SVMLightWrapper{
 private:
  DOC** docs; // these are the training examples
  long totwords, totdoc, i;
  double* target;
  double* alpha_in;
  KERNEL_CACHE* kernel_cache;
  MODEL* model; // these is the svm model

  //Constructor
  SVMLightWrapper(){
    //Init all variables
    alpha_in = NULL;
    kernel_cache = NULL; 
    model = (MODEL*) my_malloc(sizeof(MODEL));
    learn_param = new LEARN_PARM;
    kernel_param = new KERNEL_PARM;
    
    //Init the learning parameters
    verbosity = 1;
    learn_param->alphafile[0] = ' ';
    learn_param->biased_hyperplane = 1;
    learn_param->sharedslack = 0;
    learn_param->skip_final_opt_check = 0;
    learn_param->svm_maxqpsize = 10;
    learn_param->svm_newvarsinqp = 0;
    learn_param->svm_iter_to_shrink = 2;
    learn_param->kernel_cache_size = 40;
    learn_param->maxiter = 100000;
    learn_param->svm_costratio = 1.0;
    learn_param->svm_costratio_unlab = 1.0;
    learn_param->svm_unlabbound = 1E-5;
    learn_param->eps = 0.1;
    learn_param->transduction_posratio = -1.0;
    learn_param->epsilon_crit = 0.001;
    learn_param->epsilon_a = 1E-15;
    learn_param->compute_loo = 0;
    learn_param->rho = 1.0;
    learn_param->xa_depth = 0;
    learn_param->svm_c = 0.01;// Apply the soft margin C = 0.01
    learn_param->type = REGRESSION;
    learn_param->remove_inconsistent = 0;
    
    //Init the kernel parameters
    kernel_param->rbf_gamma = 1.0;
    kernel_param->coef_lin = 1;
    kernel_param->coef_const = 1;
    kernel_param->kernel_type = LINEAR;
    kernel_param->poly_degree = 3;
  }
  
  // Destructor
  virtual ~SVMLightWrapper(){
    // Cleaning up and free memory
    if(kernel_cache)
      kernel_cache_cleanup(kernel_cache);
    free(alpha_in);
    free_model(model,0);
    for(int i = 0; i < totdoc; i++)
      free_example(docs[i],1);
    free(docs);
    free(target);
  }

 public:
  LEARN_PARM* learn_param;
  KERNEL_PARM* kernel_param;

  static SVMLightWrapper* getInstance();
  
  inline void saveModelToFile(const std::string _modelFilename, const std::string _identifier = "svmlight"){
    write_model(const_cast<char*>(_modelFilename.c_str()),model);
  }

  void loadModelFromFile(const std::string _modelFilename, const std::string _identifier = "svmlight"){
    this->model = read_model(const_cast<char*>(_modelFilename.c_str()));
  }

  void read_problem(char* filename){
    read_documents(filename, &docs, &target, &totwords, &totdoc);
  }

  void train(){
    svm_learn_regression(docs, target, totdoc, totwords, learn_param, kernel_param, &kernel_cache, model);
  }

  // Generating a single feature vector from the trained support vectors
  void getSingleDetectingVector(std::vector<float>& singleDetectorVector, std::vector<unsigned int>& singleDetectorVectorIndices){
    DOC** supveclist = model->supvec;
    printf("Calculating single descriptor vector out of the support vectors found (this may take some time...\n");
    
    singleDetectorVector.clear();
    singleDetectorVector.resize(model->totwords + 1,0.);
    printf("Resulting vector size: %lu\n",singleDetectorVector.size());

    // Iterate over every support vector
    for(long ssv = 1; ssv < model->sv_num; ++ssv){
      DOC* singleSupportVector = supveclist[ssv];
      SVECTOR* singleSupportVectorValues = singleSupportVector->fvec;
      WORD singleSupportVectorComponent;
      
      // Iterate over the components of the support vector and populate the detector vector
      for(unsigned long singleFeature = 0; singleFeature < model->totwords; ++singleFeature){
	singleSupportVectorComponent = singleSupportVectorValues->words[singleFeature];
	singleDetectorVector.at(singleSupportVectorComponent.wnum) += (singleSupportVectorComponent.weight * model->alpha[ssv]);
      }
    }
    
    singleDetectorVector.at(model->totwords) = -model->b;
  }
};

//Singleton
SVMLightWrapper* SVMLightWrapper::getInstance(){
  static SVMLightWrapper theInstance;
  return &theInstance;
}

#endif  /*SVMLightWrapper*/
