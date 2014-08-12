#include <stdio.h>
#include <algorithm> 
#include <iostream>
#include <dirent.h>
#include <fstream>
#include <glog/logging.h>
#include <tinyxml2.h>

#include <opencv2/opencv.hpp>

#include "svm_wrapper.h"

using namespace std;
using namespace cv;
using namespace tinyxml2;

//Parameter Definitions
static string sampleListPath = "/Users/david/Documents/Development/VOC2007/VOCdevkit/VOC2007/ImageSets/Main/";
static string sampleImagesPath = "/Users/david/Documents/Development/VOC2007/VOCdevkit/VOC2007/JPEGImages/";
static string annotationsPath = "/Users/david/Documents/Development/VOC2007/VOCdevkit/VOC2007/Annotations/";
static string featuresFile = "/Volumes/EXTERNAL/DISSERTATION/MODELS/PASCAL/";
static string svmModelFile = "/Volumes/EXTERNAL/DISSERTATION/MODELS/PASCAL/";
//static string descriptorVectorFile = "/Users/david/Documents/HOGPascalTraining/genfiles/descriptorVector.dat";
static string category;

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

static void getROI(const string& filename, vector<Rect>& posRegions, vector<Rect>& negRegions, bool c){
  if(c == true){
    vector<string> parts = split(filename,"/");
    string imageName = split(parts[parts.size()-1],".")[0];
    string annPath =  annotationsPath+imageName+".xml";
    XMLDocument doc;
    doc.LoadFile(annPath.c_str());

    //cout << "Retrieving annotations from " << annPath << endl;
    XMLNode* root = doc.FirstChildElement("annotation");
    for(XMLNode* child = root->FirstChildElement(); child != 0; child = child->NextSibling()){
      //Find the object node in the annotations
      if(strcmp(child->Value(),"object") == 0){
        for(XMLNode* object = child->FirstChildElement(); object != 0; object = object->NextSibling()){
          // Check if the name correspond to the category
          if(strcmp(object->Value(),"name")==0){
            if(strcmp(object->ToElement()->GetText(),category.c_str())==0){
              //Retrieving the bounding box from the annotations
              int xmin = atoi(object->Parent()->FirstChildElement("bndbox")->FirstChildElement("xmin")->GetText());
              int ymin = atoi(object->Parent()->FirstChildElement("bndbox")->FirstChildElement("ymin")->GetText());
              int xmax = atoi(object->Parent()->FirstChildElement("bndbox")->FirstChildElement("xmax")->GetText());
              int ymax = atoi(object->Parent()->FirstChildElement("bndbox")->FirstChildElement("ymax")->GetText());

              //cout << "Region found: " << xmin << " - " << ymin << " - " << xmax << " - " << ymax << endl;

              posRegions.push_back(Rect(xmin,ymin,xmax-xmin,ymax-ymin));
            }
            else{
              //Retrieving the bounding box from the annotations
              int xmin = atoi(object->Parent()->FirstChildElement("bndbox")->FirstChildElement("xmin")->GetText());
              int ymin = atoi(object->Parent()->FirstChildElement("bndbox")->FirstChildElement("ymin")->GetText());
              int xmax = atoi(object->Parent()->FirstChildElement("bndbox")->FirstChildElement("xmax")->GetText());
              int ymax = atoi(object->Parent()->FirstChildElement("bndbox")->FirstChildElement("ymax")->GetText());

              //cout << "Region found: " << xmin << " - " << ymin << " - " << xmax << " - " << ymax << endl;

              negRegions.push_back(Rect(xmin,ymin,xmax-xmin,ymax-ymin));
            }
          }
        }
      }
    }
  }
  else{
    // Add random negative patches to the training (HARD NEGATIVES)
    Mat im = imread(filename,0);
    //Generate 10 random image
    for(int i = 0; i < 10; ++i){
      int range_x = im.cols - 64;
      int range_y = im.rows - 128;

      int x = 0;
      int y = 0;
      Rect r;
      if(range_x < 0){
        y = rand() % (im.rows-128);
        r.x = x;
        r.y = y;
        r.width = im.cols;
        r.height = 128;
      }
      else if(range_y < 0){
        x = rand() % (im.cols-128);
        r.x = x;
        r.y = y;
        r.width = 64;
        r.height = im.rows;
      }
      else{
        x = rand() % (im.cols-64);
        y = rand() % (im.rows-128);
        r.x = x;
        r.y = y;
        r.width = 64;
        r.height = 128;
      }
      
      negRegions.push_back(r);

      // imshow("Original",im);
      // imshow("Random Cut",im(r));
      // waitKey(0);
    }
    im.release();
  }
}

static void getSamples(const string& sampleListPath, vector<string>& posFilenames, vector<string>& negFilenames){
  LOG(INFO) << "Opening list: " << sampleListPath;
  string separator = " ";
  fstream File;
  File.open(sampleListPath.c_str(), ios::in);
  if(File.good() && File.is_open()){
    string line;
    while(getline(File,line)){
      vector<string> tokens = split(line,separator);
      //cout << sampleImagesPath+tokens[0]+".jpg" << endl;
      if(tokens[1] != "-1")
        posFilenames.push_back(sampleImagesPath+tokens[0]+".jpg");
      else
        negFilenames.push_back(sampleImagesPath+tokens[0]+".jpg");
    }
    File.close();
  }
}

static void calculateFeaturesFromInput(const string& imageFilename, HOGDescriptor& hog, LibSVM::SVMTrainer& svm, bool c){
  cout << imageFilename << " - " << c << endl;
  Mat imageData = imread(imageFilename, 0);

  cout << imageData.rows << " - " << imageData.cols << endl;
  vector<Rect> posRegions;
  vector<Rect> negRegions;

  //Find the bounding box (ROI) in the annotations and cut the image
  getROI(imageFilename, posRegions, negRegions, c);

  LOG(INFO) << " pos size: " << posRegions.size() << " neg size: " << negRegions.size();

  for(int roi = 0; roi < posRegions.size(); ++roi){
    //LOG(INFO) << "Positive Patch Found";
    vector<float> featureVector;

    Mat selection = imageData(posRegions[roi]);
    resize(selection,selection,hog.winSize);

    //Check a valid image input
    if(selection.empty()){
      featureVector.clear();
      LOG(ERROR) << "Error: HOG image " << imageFilename.c_str()  <<  " is empty, features calculation skipped!";
      return;
    }

    //Check a valid image size
    if(selection.cols != hog.winSize.width || selection.rows != hog.winSize.height){
      featureVector.clear();
      LOG(ERROR) << "Error: Image " << imageFilename.c_str() << " dimensions (" << selection.cols 
        << "x" << selection.rows << ") do not match HOG window size (" << hog.winSize.width 
        << "x" << hog.winSize.height << ")";
    }

    hog.compute(selection,featureVector,Size(8,8),Size(0,0));
    //LOG(INFO) << "Computed HOG features for patch";
    svm.writeFeatureVectorToFile(featureVector,true); 
    selection.release();
    featureVector.clear();
  }
  for(int roi = 0; roi < negRegions.size(); ++roi){
    //LOG(INFO) << "Negative Patch Found";
    vector<float> featureVector;
    //cout << negRegions[roi].tl() << " - " << negRegions[roi].br() << endl;
    
    Mat selection = imageData(negRegions[roi]);
    resize(selection,selection,hog.winSize);

    //Check a valid image input
    if(selection.empty()){
      featureVector.clear();
      LOG(ERROR) << "Error: HOG image " << imageFilename.c_str()  <<  " is empty, features calculation skipped!";
      return;
    }

    //Check a valid image size
    if(selection.cols != hog.winSize.width || selection.rows != hog.winSize.height){
      featureVector.clear();
      LOG(ERROR) << "Error: Image " << imageFilename.c_str() << " dimensions (" << selection.cols 
        << "x" << selection.rows << ") do not match HOG window size (" << hog.winSize.width 
        << "x" << hog.winSize.height << ")";
    }

    hog.compute(selection,featureVector,Size(8,8),Size(0,0));
    //LOG(INFO) << "Computed patch HOG features";
    svm.writeFeatureVectorToFile(featureVector,false); 
    selection.release();
    featureVector.clear();
  }
  imageData.release();              // we don't need the original image anymore
}

int main(int argc, char** argv){
  category = argv[1];
  
  transform(category.begin(),category.end(),category.begin(), ::toupper);
  //Define the folders
  featuresFile += category + "/HOG-LINEAR_SVM/features.dat";
  svmModelFile += category + "/HOG-LINEAR_SVM/svmModel.dat";

  HOGDescriptor hog;
  hog.winSize = Size(64,128);
  
  //Get the positive and negative training samples
  static vector<string> positiveTrainingImages;
  static vector<string> negativeTrainingImages;
  static vector<string> validExtensions;
  
  //Get the images found in the list for the category
  transform(category.begin(),category.end(),category.begin(),::tolower);
  sampleListPath = sampleListPath+category+"_train.txt";
  getSamples(sampleListPath, positiveTrainingImages, negativeTrainingImages);

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
  
  LOG(INFO) << "Generating the HOG features and saving it in file "<< featuresFile.c_str();
  float percent;

  LibSVM::SVMTrainer svm(featuresFile.c_str());
  //Iterate over sample images
  for(unsigned long currentFile = 0; currentFile < overallSamples; ++currentFile){
    storeCursor();
    const string currentImageFile = currentFile < positiveTrainingImages.size() ? positiveTrainingImages.at(currentFile): negativeTrainingImages.at(currentFile-positiveTrainingImages.size());
    // Output progress
    if((currentFile+1)%1 == 0 || currentFile+1 == overallSamples){
      percent = ((currentFile+1)*100)/overallSamples;
      char progressString[500];
      sprintf(progressString,"%5lu (%3.0f%%): File'%s'\n",(currentFile+1),percent,currentImageFile.c_str());
      LOG(INFO) << progressString;
      fflush(stdout);
      resetCursor();
    }
    
    // Calculate feature vector for the current image file
    bool c = false;
    if(currentFile < positiveTrainingImages.size())
      c = true;
    
    // Calculate feature vector for the current image file
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
  LOG(INFO) << "SVM Model saved to " << svmModelFile << endl;

  return 0;
}
