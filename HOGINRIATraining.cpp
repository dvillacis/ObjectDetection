#include <stdio.h>
#include <iostream>
#include <dirent.h>
#include <fstream>

#include <tinyxml2.h>

#include <opencv2/opencv.hpp>

#include "svm_light_wrapper.h"

using namespace std;
using namespace cv;
using namespace tinyxml2;

//Parameter Definitions
static string sampleListPath = "/Users/david/Documents/Development/INRIAPerson/train_64x128_H96/";
static string testPath = "/Users/david/Documents/Development/INRIAPerson/Test/";
static string featuresFile = "/Users/david/Documents/HOGINRIATraining/genfiles/features.dat";
static string svmModelFile = "/Users/david/Documents/HOGINRIATraining/genfiles/svmModel.dat";
static string descriptorVectorFile = "/Users/david/Documents/HOGINRIATraining/genfiles/descriptorVector.dat";

//HOG Training parameters
static const Size trainingPadding = Size(0,0);
static const Size winStride = Size(8,8);

// Helper Functions
static void storeCursor(){
  printf("\033[s");
}

static void resetCursor(){
  printf("\033[u");
}

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

static void getSamples(string& listPath, vector<string>& posFilenames, vector<string>& negFilenames, const vector<string>& validExtensions){
  string posPath = listPath+"pos/";
  string negPath = listPath+"neg/";
  cout << "Getting positive files in: " << sampleListPath << endl;
  struct dirent* ep;
  DIR* dp = opendir(posPath.c_str());
  if(dp!=NULL){
    int i = 0;
    while((ep = readdir(dp))){
      i++;
      if(i == 100)
        break;
      if(ep->d_type & DT_DIR){
        continue;
      }
      size_t extensionLocation = string(ep->d_name).find_last_of(".");
      string tempExt = string(ep->d_name).substr(extensionLocation+1);
      if(find(validExtensions.begin(),validExtensions.end(), tempExt)!= validExtensions.end()){
        posFilenames.push_back((string)posPath + ep->d_name);
        cout << "Adding " << (string)posPath + ep->d_name << " to the positive training samples" << endl;
      }
    }
  }
  cout << "Getting negative files in: " << sampleListPath << endl;
  dp = opendir(negPath.c_str());
  if(dp!=NULL){
    int i = 0;
    while((ep = readdir(dp))){
      i++;
      if(i == 100)
        break;
      if(ep->d_type & DT_DIR){
        continue;
      }
      size_t extensionLocation = string(ep->d_name).find_last_of(".");
      string tempExt = string(ep->d_name).substr(extensionLocation+1);
      if(find(validExtensions.begin(),validExtensions.end(), tempExt)!= validExtensions.end()){
        negFilenames.push_back((string)negPath + ep->d_name);
        cout << "Adding " << (string)negPath + ep->d_name << " to the negative training samples" << endl;
      }
    }
  }
  return;
}

static void calculateFeaturesFromInput(const string& imageFilename, HOGDescriptor& hog, SVMLight::SVMTrainer& svm, bool c){
  Mat imageData = imread(imageFilename, CV_LOAD_IMAGE_GRAYSCALE);
  //resize(imageData,imageData,hog.winSize);
  vector<float> featureVector;
  hog.compute(imageData,featureVector,Size(8,8),Size(0,0));
  //cout << "\t\tComputed patch HOG features" << endl;
  svm.writeFeatureVectorToFile(featureVector,c); 
  imageData.release();
  featureVector.clear();
  //cout << "\t\tClean exit" << endl; 
}

static void showDetections(const vector<Rect>& found, Mat& imageData){
  vector<Rect> found_filtered;
  size_t i, j;
  for (i = 0; i < found.size(); ++i) {
    Rect r = found[i];
    for (j = 0; j < found.size(); ++j)
      if (j != i && (r & found[j]) == r)
  break;
    if (j == found.size())
      found_filtered.push_back(r);
  }
  for (i = 0; i < found_filtered.size(); i++) {
    Rect r = found_filtered[i];
    rectangle(imageData, r.tl(), r.br(), Scalar(64, 255, 64), 3);
  }

  cout << "Instances found: " << found_filtered.size() << endl;
} 

static void detectTest(const HOGDescriptor& hog, Mat& imageData){
  vector<Rect> found;
  int groupThreshold = 2;
  Size padding(Size(32,32));
  Size winStride(Size(8,8));
  double hitThreshold = 0.;
  hog.detectMultiScale(imageData,found, hitThreshold,winStride,padding,1.05,groupThreshold);
  showDetections(found, imageData);
}

static void test(){
  HOGDescriptor hog;
  hog.winSize = Size(64,128);
  SVMLight::SVMClassifier classifier(svmModelFile);
  cout << "Getting the Classifier" << endl;
  vector<float> descriptorVector = classifier.getDescriptorVector();
  cout << "Getting the Descriptor Vector" << endl;
  hog.setSVMDetector(descriptorVector);
  cout << "Applied the descriptor to the hog class" << endl;
  //hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

  vector<string> testImagesPos;
  vector<string> testImagesNeg;
  static vector<string> validExtensions;
  validExtensions.push_back("jpg");
  validExtensions.push_back("png");

  getSamples(testPath, testImagesPos, testImagesNeg, validExtensions);
  for(int i = 0; i < testImagesPos.size(); ++i){
    cout << "Testing image: " << testImagesPos[i] << endl;
    Mat image = imread(testImagesPos[i],CV_LOAD_IMAGE_GRAYSCALE);
    detectTest(hog,image);
    imshow("HOG Custom Detection",image);
    waitKey(0);
  }
}

int main(int argc, char** argv){
  HOGDescriptor hog;
  hog.winSize = Size(64,128);
  
  //Get the positive and negative training samples
  static vector<string> positiveTrainingImages;
  static vector<string> negativeTrainingImages;
  static vector<string> validExtensions;
  validExtensions.push_back("jpg");
  validExtensions.push_back("png");
  
  //Get the images in the training folders
  getSamples(sampleListPath, positiveTrainingImages, negativeTrainingImages, validExtensions);

  //Count the total number of samples
  unsigned long overallSamples = positiveTrainingImages.size()+negativeTrainingImages.size();
  
  //Make sure there are samples to train
  if(overallSamples == 0){
    cout << "No training samples found, exiting..." << endl;
    return -1;
  }

  //Setting the locale
  setlocale(LC_ALL, "C");
  setlocale(LC_NUMERIC, "C");
  setlocale(LC_ALL,"POSIX");
  
  cout << "Generating the HOG features and saving it in file "<< featuresFile.c_str() << endl;
  float percent;

  SVMLight::SVMTrainer svm(featuresFile.c_str());
  //Iterate over sample images
  for(unsigned long currentFile = 0; currentFile < overallSamples; ++currentFile){
    storeCursor();
    const string currentImageFile = currentFile < positiveTrainingImages.size() ? positiveTrainingImages.at(currentFile): negativeTrainingImages.at(currentFile-positiveTrainingImages.size());
    
    // Output progress
    if((currentFile+1)%1 == 0 || currentFile+1 == overallSamples){
      percent = ((currentFile+1)*100)/overallSamples;
      printf("%5lu (%3.0f%%):\t File'%s'\n",(currentFile+1),percent,currentImageFile.c_str());
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
  cout << "Finished writing the features" << endl;

  // Starting the training of the model
  cout << "Starting the training of the model using SVMLight" << endl;
  svm.trainAndSaveModel(svmModelFile);
  cout << "SVM Model saved to " << svmModelFile << endl;
  
  // Test the just trained descriptor
  test();

  return 0;
}
