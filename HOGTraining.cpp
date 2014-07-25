#include <stdio.h>
#include <iostream>
#include <dirent.h>
#include <fstream>

#include <opencv2/opencv.hpp>

#include "svm_light_wrapper.h"

using namespace std;
using namespace cv;

//Parameter Definitions
static string posSamplesDir = "/Users/david/Documents/Development/INRIAPerson/train_64x128_H96/pos/";
static string negSamplesDir = "/Users/david/Documents/Development/INRIAPerson/train_64x128_H96/neg/";
static string featuresFile = "/Users/david/Documents/HOGTraining/genfiles/features.dat";
static string svmModelFile = "/Users/david/Documents/HOGTraining/genfiles/svmModel.dat";
static string descriptorVectorFile = "/Users/david/Documents/HOGTraining/genfiles/descriptorVector.dat";

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

static void getFilesInDirectory(const string& dirName, vector<string>& fileNames, const vector<string>& validExtensions){
  cout << "Opening directory: " << dirName << endl;
  struct dirent* ep;
  size_t extensionLocation;
  DIR* dp = opendir(dirName.c_str());
  if(dp != NULL){
    while((ep = readdir(dp))){
      if(ep->d_type & DT_DIR)
	continue;
      extensionLocation = string(ep->d_name).find_last_of(".");
      // Find matching extensions
      string tempExt = string(ep->d_name).substr(extensionLocation + 1);
      transform(tempExt.begin(),tempExt.end(),tempExt.begin(),::tolower);
      if(find(validExtensions.begin(),validExtensions.end(), tempExt) != validExtensions.end()){
	cout << "Found matching data file: " << ep->d_name << endl;
	fileNames.push_back((string) dirName + ep->d_name);
      }
      else
	cout << "Found file does not match the requiered file extension, skipping: " << ep->d_name << endl;
    }
    (void) closedir(dp);
  }
  else
    cout << "Error opening directory: " << dirName.c_str() << endl;
  return;
}

static void calculateFeaturesFromInput(const string& imageFilename, vector<float>& featureVector, HOGDescriptor& hog){
  Mat imageData = imread(imageFilename, 0);
  
  //Resize the image to match the hog size
  Size size(64,128);
  resize(imageData,imageData,size);

  //Check a valid image input
  if(imageData.empty()){
    featureVector.clear();
    cout << "Error: HOG image " << imageFilename.c_str()  <<  " is empty, features calculation skipped!" << endl;
    return;
  }

  //Check a valid image size
  if(imageData.cols != hog.winSize.width || imageData.rows != hog.winSize.height){
    featureVector.clear();
    cout << "Error: Image " << imageFilename.c_str() << " dimensions (" << imageData.cols 
	 << "x" << imageData.rows << ") do not match HOG window size (" << hog.winSize.width 
	 << "x" << hog.winSize.height << ")" << endl;
  }
  vector<Point> locations;
  hog.compute(imageData, featureVector, winStride, trainingPadding, locations);
  imageData.release();
}

static void saveDescriptorVectorToFile(vector<float>& descriptorVector, vector<unsigned int>& _vectorIndices, string filename){
  cout << "Saving descriptor vector to file " << filename.c_str() << endl;
  string separator = " ";
  fstream File;
  float percent;
  File.open(filename.c_str(), ios::out);
  if(File.good() && File.is_open()){
    cout << "Saving descriptor vector features: \t" << endl;
    storeCursor();
    for(int feature = 0; feature < descriptorVector.size(); ++feature){
      if((feature%300 == 0)||(feature == (descriptorVector.size()-1))){
	percent = ((1+feature)*100)/descriptorVector.size();
	printf("%4u (%3.0f%%)...",feature,percent);
	fflush(stdout);
	resetCursor();
      }
      File << descriptorVector.at(feature) << separator;
    }
    printf("\n");
    File << endl;
    File.flush();
    File.close();
  }
  else
    cout << "Error writing the file " << filename.c_str() << endl;
}

int main(int argc, char** argv){
  HOGDescriptor hog;
  
  //Get the positive and negative training samples
  static vector<string> positiveTrainingImages;
  static vector<string> negativeTrainingImages;
  static vector<string> validExtensions;
  
  //Set the valid extension to known images formats
  validExtensions.push_back("jpg");
  validExtensions.push_back("png");
  validExtensions.push_back("ppm");
  
  //Get the images found in the directories
  getFilesInDirectory(posSamplesDir, positiveTrainingImages, validExtensions);
  getFilesInDirectory(negSamplesDir, negativeTrainingImages, validExtensions);

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

  fstream File;
  File.open(featuresFile.c_str(),ios::out);
  if(File.good() && File.is_open()){
    //Iterate over sample images
    for(unsigned long currentFile = 0; currentFile < overallSamples; ++currentFile){
      storeCursor();
      vector<float> featureVector;
      const string currentImageFile = 
	currentFile < positiveTrainingImages.size() ? positiveTrainingImages.at(currentFile):
	negativeTrainingImages.at(currentFile-positiveTrainingImages.size());
      
      // Output progress
      if((currentFile+1)%1 == 0 || currentFile+1 == overallSamples){
	percent = ((currentFile+1)*100)/overallSamples;
	printf("%5lu (%3.0f%%):\t File'%s'\n",(currentFile+1),percent,currentImageFile.c_str());
	fflush(stdout);
	resetCursor();
      }
      
      // Calculate feature vector for the current image file
      calculateFeaturesFromInput(currentImageFile, featureVector, hog);
      if(!featureVector.empty()){
	// Add a class label depending on the source folder
	File << ((currentFile < positiveTrainingImages.size())?"+1":"-1");
	
	// Save the feature vector components
	for(unsigned int feature = 0; feature < featureVector.size(); ++feature){
	  File << " " << (feature+1) << ":" << featureVector.at(feature);
	}
	File << endl;
      }
    }
    printf("\n");
    File.flush();
    File.close();
  }
  else{
    cout << "Error opening file: " << featuresFile.c_str() << endl;
    return -1;
  }

  // Starting the training of the model
  cout << "Starting the training of the model using SVMLight" << endl;
  SVMLightWrapper::getInstance()->read_problem(const_cast<char*>(featuresFile.c_str()));
  SVMLightWrapper::getInstance()->train();
  cout << "Model trained" << endl;
  SVMLightWrapper::getInstance()->saveModelToFile(svmModelFile);

  // Getting a representative HOG feature vector using the support vectors obtained
  vector<float> descriptorVector;
  vector<unsigned int> descriptorVectorIndices;
  SVMLightWrapper::getInstance()->getSingleDetectingVector(descriptorVector, descriptorVectorIndices);
  saveDescriptorVectorToFile(descriptorVector, descriptorVectorIndices, descriptorVectorFile);
  return 0;
}
