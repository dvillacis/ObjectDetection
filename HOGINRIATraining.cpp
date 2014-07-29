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
static string sampleListPath = "/Users/david/Documents/Development/INRIAPerson/Train/";
static string trainAnnotationsPath = "/Users/david/Documents/Development/INRIAPerson/Train/annotations/";
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
  cout << "Getting positive files in: " << posPath << endl;
  struct dirent* ep;
  DIR* dp = opendir(posPath.c_str());
  if(dp!=NULL){
    int i = 0;
    while((ep = readdir(dp))){
      // i++;
      // if(i == 100)
      //   break;
      if(ep->d_type & DT_DIR){
        continue;
      }
      size_t extensionLocation = string(ep->d_name).find_last_of(".");
      string tempExt = string(ep->d_name).substr(extensionLocation+1);
      if(find(validExtensions.begin(),validExtensions.end(), tempExt)!= validExtensions.end()){
        posFilenames.push_back((string)posPath + ep->d_name);
        //cout << "Adding " << (string)posPath + ep->d_name << " to the positive training samples" << endl;
      }
    }
  }
  cout << "Getting negative files in: " << negPath << endl;
  dp = opendir(negPath.c_str());
  if(dp!=NULL){
    int i = 0;
    while((ep = readdir(dp))){
      // i++;
      // if(i == 80)
      //   break;
      if(ep->d_type & DT_DIR){
        continue;
      }
      size_t extensionLocation = string(ep->d_name).find_last_of(".");
      string tempExt = string(ep->d_name).substr(extensionLocation+1);
      if(find(validExtensions.begin(),validExtensions.end(), tempExt)!= validExtensions.end()){
        negFilenames.push_back((string)negPath + ep->d_name);
        //cout << "Adding " << (string)negPath + ep->d_name << " to the negative training samples" << endl;
      }
    }
  }
  return;
}

static string getAnnotation(const string& imageFilename, const string& annotationsPath){
  //Get the annotation filename
  vector<string> parts = split(imageFilename,"/");
  string imageName = split(parts[parts.size()-1],".")[0];
  string annPath = annotationsPath+imageName+".txt";
  return annPath;
}

static vector<Rect> getROI(const string& imageFilename, bool c){
  vector<Rect> bboxes;
  if(c == true){
    string annPath = getAnnotation(imageFilename,trainAnnotationsPath);

    fstream annFile;
    annFile.open(annPath.c_str(),ios::in);
    if(annFile.good() && annFile.is_open()){
      string line;
      while(getline(annFile,line)){
        if(line.find("Bounding box for object") != string::npos){
          string temp = split(line," : ")[1];
          vector<string> coords = split(temp," - ");
          int xmin = atoi(split(coords[0],", ")[0].erase(0,1).c_str());
          int ymin = atoi(split(coords[0],", ")[1].c_str());
          int xmax = atoi(split(coords[1],", ")[0].erase(0,1).c_str());
          int ymax = atoi(split(coords[1],", ")[1].c_str());
          bboxes.push_back(Rect(xmin,ymin,xmax-xmin,ymax-ymin));
        }
      }
      annFile.close();
    }
    else
      cout << "Couldnt open annotations file for: " << annPath << endl;
  }
  else
  {
    //TODO: get the random images from the negative input
    Mat im = imread(imageFilename,0);
    bboxes.push_back(Rect(0,0,im.cols,im.rows));
  }
  
  return bboxes;
}

static void calculateFeaturesFromInput(const string& imageFilename, HOGDescriptor& hog, SVMLight::SVMTrainer& svm, bool c){
  Mat imageData = imread(imageFilename, CV_LOAD_IMAGE_GRAYSCALE);

  //Finding person bounding boxes inside the image
  vector<Rect> ROI = getROI(imageFilename,c);
  for(int i = 0; i < ROI.size(); ++i){
    Mat patch = imageData(ROI[i]);
    // imshow("Training Image (+)",patch);
    // waitKey(0);

    resize(patch,patch,hog.winSize);
    vector<float> featureVector;
    hog.compute(patch,featureVector,Size(8,8),Size(0,0));
    svm.writeFeatureVectorToFile(featureVector,c); 
    patch.release();
    featureVector.clear();
  }
  imageData.release();
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
  
  return 0;
}
