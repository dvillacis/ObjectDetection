#include <stdio.h>
#include <iostream>
#include <dirent.h>
#include <fstream>

#include <tinyxml2.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include "svm_wrapper.h"
#include "INRIATestingUtils.h"

using namespace std;
using namespace cv;
using namespace tinyxml2;
using namespace boost;

//Parameter Definitions
static string sampleListPath = "/Users/david/Documents/Development/INRIAPerson/Train/";
static string trainAnnotationsPath = "/Users/david/Documents/Development/INRIAPerson/Train/annotations/";
static string featuresFile = "/Volumes/EXTERNAL/DISSERTATION/MODELS/INRIA/";
static string svmModelFile = "/Volumes/EXTERNAL/DISSERTATION/MODELS/INRIA/";
//static string logDir = "/Volumes/EXTERNAL/DISSERTATION/MODELS/INRIA/";
//static string descriptorVectorFile = "/Users/david/Documents/HOGINRIATraining/genfiles/descriptorVector.dat";

//HOG Training parameters
static const Size trainingPadding = Size(0,0);
static const Size winStride = Size(8,8);

//Utilities
static INRIAUtils::INRIATestingUtils* utils = new INRIAUtils::INRIATestingUtils();

// Helper Functions
static void storeCursor(){
  printf("\033[s");
}

static void resetCursor(){
  printf("\033[u");
}

static void calculateFeaturesFromInput(const string& imageFilename, HOGDescriptor& hog, LibSVM::SVMTrainer& svm, bool c){
  Mat imageData = imread(imageFilename, CV_LOAD_IMAGE_COLOR);

  //Finding person bounding boxes inside the image
  utils->setAnnotationsPath(trainAnnotationsPath);
  vector<Rect> ROI = utils->getROI(imageFilename,c);
  for(int i = 0; i < ROI.size(); ++i){
    Mat patch = imageData(ROI[i]);
    resize(patch,patch,hog.winSize);
    vector<float> featureVector;
    hog.compute(patch,featureVector,winStride,trainingPadding);
    svm.writeFeatureVectorToFile(featureVector,c); 

    //Add the reflection of the patch
    Mat reflectedPatch;
    flip(patch, reflectedPatch,1);
    featureVector.clear();
    hog.compute(reflectedPatch,featureVector,winStride,trainingPadding);
    svm.writeFeatureVectorToFile(featureVector,c);

    patch.release();
    reflectedPatch.release();
    featureVector.clear();
  }
  imageData.release();
}

static void hardNegativeTraining(string& svmModelFile, HOGDescriptor& hog, LibSVM::SVMTrainer& svm){
  LibSVM::SVMClassifier classifier(svmModelFile);
  vector<float> descriptorVector = classifier.getDescriptorVector();
  hog.setSVMDetector(descriptorVector);

  //Get the positive and negative training samples
  vector<string> positiveTrainingImages;
  vector<string> negativeTrainingImages;
  vector<string> validExtensions;
  validExtensions.push_back(".jpg");
  validExtensions.push_back(".png");
  
  //Get the negavive images in the training folders
  utils->setAnnotationsPath(trainAnnotationsPath);
  utils->getSamples(sampleListPath+"neg/", negativeTrainingImages, validExtensions);
  
  //Make sure there are samples to train
  if(negativeTrainingImages.size() == 0){
    LOG(ERROR) << "No training samples found, exiting...";
    return;
  }

  for(int i = 0; i < negativeTrainingImages.size(); ++i){
    vector<Rect> found;
    utils->setAnnotationsPath(trainAnnotationsPath);
    utils->getDetections(negativeTrainingImages[i], hog, found,false);

    Mat originalImage = imread(negativeTrainingImages[i],CV_LOAD_IMAGE_COLOR);

    for(int j = 0; j < found.size(); ++j){
      vector<float> featureVector;
      
      Rect originalRect(0,0,originalImage.cols,originalImage.rows);
      Rect intersection = originalRect & found[j];
      Mat patch = originalImage(intersection);

      resize(patch,patch,Size(64,128));
      hog.compute(patch,featureVector,winStride,trainingPadding);
      svm.writeFeatureVectorToFile(featureVector,false);
      patch.release();
      featureVector.clear();
    }
    originalImage.release();
  }
}

int main(int argc, char** argv){

  string kernelString = "LINEAR";
  featuresFile += "HOG-"+kernelString+"_SVM/features.dat";
  svmModelFile += "HOG-"+kernelString+"_SVM/svmModel.dat";
  //logDir += "HOG-"+kernelString+"_SVM/log/";

  //Starting the logging library
  //FLAGS_log_dir = logDir;
  FLAGS_logtostderr = false;
  FLAGS_stderrthreshold = 0;
  google::InitGoogleLogging(argv[0]);

  HOGDescriptor hog;
  hog.winSize = Size(64,128);
  
  //Get the positive and negative training samples
  vector<string> positiveTrainingImages;
  vector<string> negativeTrainingImages;
  vector<string> validExtensions;
  validExtensions.push_back(".jpg");
  validExtensions.push_back(".png");
  
  //Get the positive images in the training folders
  utils->setAnnotationsPath(trainAnnotationsPath);
  utils->getSamples(sampleListPath+"pos/", positiveTrainingImages, validExtensions);
  //Get the negavive images in the training folders
  utils->getSamples(sampleListPath+"neg/", negativeTrainingImages, validExtensions);

  //Count the total number of samples
  unsigned long overallSamples = positiveTrainingImages.size()+negativeTrainingImages.size();
  
  //Make sure there are samples to train
  if(overallSamples == 0){
    LOG(ERROR) << "No training samples found, exiting...";
    return -1;
  }

  //Setting the locale
  setlocale(LC_ALL, "C");
  setlocale(LC_NUMERIC, "C");
  setlocale(LC_ALL,"POSIX");
  
  LOG(INFO) << "Generating the HOG features and saving it in file "<< featuresFile.c_str() << endl;
  float percent;

  LibSVM::SVMTrainer svm(featuresFile.c_str());
  //Iterate over sample images
  for(unsigned long currentFile = 0; currentFile < overallSamples; ++currentFile){
    storeCursor();
    const string currentImageFile = currentFile < positiveTrainingImages.size() ? positiveTrainingImages.at(currentFile): negativeTrainingImages.at(currentFile-positiveTrainingImages.size());
    
    // Output progress
    if((currentFile+1)%1 == 0 || currentFile+1 == overallSamples){
      percent = ((currentFile+1)*100)/overallSamples;
      char progressString[50];
      sprintf(progressString,"%5lu (%3.0f%%): File'%s'\n",(currentFile+1),percent,currentImageFile.c_str());
      LOG(INFO) << progressString;
      fflush(stdout);
      resetCursor();
    }
    
    // Calculate feature vector for the current image file
    bool c = false;
    if(currentFile < positiveTrainingImages.size())
      c = true;
    calculateFeaturesFromInput(currentImageFile, hog, svm, c);
    
    //os.clear(); os.seekp(0);    // reset string stream

  }
  printf("\n");
  LOG(INFO) << "Finished writing the features";
  // Starting the training of the model
  LOG(INFO) << "Starting the training of the model using LibSVM";

  // Defining the parameters for the SVM
  CvSVMParams* myParams = new CvSVMParams(
    CvSVM::C_SVC,   // Type of SVM; using N classes here
    CvSVM::LINEAR,  // Kernel type
    3,              // Param (degree) for poly kernel only
    1.0,            // Param (gamma) for poly/rbf kernel only
    1.0,            // Param (coef0) for poly/sigmoid kernel only
    0.01,           // SVM optimization param C
    0,              // SVM optimization param nu (not used for N class SVM)
    0,              // SVM optimization param p (not used for N class SVM)
    NULL,           // class weights (or priors)
    cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 0.000001)
  );

  svm.trainAndSaveModel(svmModelFile,myParams);
  LOG(WARNING) << "SVM Model saved to " << svmModelFile;

  //Calculate the false positive to perform a hard negative training
  LOG(INFO) << "Detecting false positives in traning set to perform HARD NEGATIVE TRAINING";
  hardNegativeTraining(svmModelFile, hog, svm);
  svm.closeFeaturesFile();

  printf("\n");
  LOG(INFO) << "Finished writing the hard negative features";
  // Starting the training of the model
  LOG(INFO) << "Starting the re - training of the model using LibSVM";
  svm.trainAndSaveModel(svmModelFile,myParams);
  LOG(WARNING) << "SVM Model saved to " << svmModelFile;
  
  return 0;
}







