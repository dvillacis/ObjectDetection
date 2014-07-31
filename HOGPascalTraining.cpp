#include <stdio.h>
#include <iostream>
#include <dirent.h>
#include <fstream>

#include <tinyxml2.h>

#include <opencv2/opencv.hpp>

#include "lib_svm_wrapper.h"

using namespace std;
using namespace cv;
using namespace tinyxml2;

extern "C" 
{
    #include "svm.h" 
}

//Parameter Definitions
static string sampleListPath = "/Users/david/Documents/Development/VOC2007/VOCdevkit/VOC2007/ImageSets/Main/";
static string sampleImagesPath = "/Users/david/Documents/Development/VOC2007/VOCdevkit/VOC2007/JPEGImages/";
static string annotationsPath = "/Users/david/Documents/Development/VOC2007/VOCdevkit/VOC2007/Annotations/";
static string featuresFile = "/Users/david/Documents/HOGPascalTraining/genfiles/features.dat";
static string svmModelFile = "/Users/david/Documents/HOGPascalTraining/genfiles/svmModel.dat";
static string descriptorVectorFile = "/Users/david/Documents/HOGPascalTraining/genfiles/descriptorVector.dat";
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

static void getROI(const string& filename, vector<Rect>& posRegions, vector<Rect>& negRegions){

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

static void getSamples(const string& sampleListPath, vector<string>& posFilenames, vector<string>& negFilenames){
  cout << "Opening list: " << sampleListPath << endl;
  string separator = " ";
  fstream File;
  File.open(sampleListPath.c_str(), ios::in);
  if(File.good() && File.is_open()){
    string line;
    while(getline(File,line)){
      vector<string> tokens = split(line,separator);
      //cout << sampleImagesPath+tokens[0]+".jpg" << endl;
      if(atof(tokens[1].c_str()) != -1)
        posFilenames.push_back(sampleImagesPath+tokens[0]+".jpg");
      else
        negFilenames.push_back(sampleImagesPath+tokens[0]+".jpg");
    }
    File.close();
  }
}

static void calculateFeaturesFromInput(const string& imageFilename, HOGDescriptor& hog, LibSVM::SVMTrainer& svm){
  Mat imageData = imread(imageFilename, 0);
  vector<Rect> posRegions;
  vector<Rect> negRegions;
  //Find the bounding box (ROI) in the annotations and cut the image
  getROI(imageFilename, posRegions, negRegions);
  for(int roi = 0; roi < posRegions.size(); ++roi){
    cout << "\t\tPositive Patch Found" << endl;
    vector<float> featureVector;
    Mat selection = imageData(posRegions[roi]);
    cout << "\t\tPatch size: " << selection.cols << " x " << selection.rows << "\t";
    resize(selection,selection,hog.winSize);
    cout << "\t\tAfter resize: " << selection.cols << " x " << selection.rows <<  endl;
    //Show the training image cutted from the original image
    // imshow(category+" positive ",selection);
    // waitKey(0); 

    //Check a valid image input
    if(selection.empty()){
      featureVector.clear();
      cout << "Error: HOG image " << imageFilename.c_str()  <<  " is empty, features calculation skipped!" << endl;
      return;
    }

    //Check a valid image size
    if(selection.cols != hog.winSize.width || selection.rows != hog.winSize.height){
      featureVector.clear();
      cout << "Error: Image " << imageFilename.c_str() << " dimensions (" << selection.cols 
        << "x" << selection.rows << ") do not match HOG window size (" << hog.winSize.width 
        << "x" << hog.winSize.height << ")" << endl;
    }

    hog.compute(selection,featureVector,Size(8,8),Size(0,0));
    cout << "\t\tComputed patch HOG features" << endl;
    svm.writeFeatureVectorToFile(featureVector,true); 
    selection.release();
    featureVector.clear();
    cout << "\t\tClean exit" << endl;
  }
  for(int roi = 0; roi < negRegions.size(); ++roi){
    cout << "\t\tNegative Patch Found" << endl;
    vector<float> featureVector;
    Mat selection = imageData(negRegions[roi]);
    cout << "\t\t ROI: " << negRegions[roi].tl() << "-" << negRegions[roi].br() << endl;
    cout << "\t\tPatch size: " << selection.cols << " x " << selection.rows << "\t";
    resize(selection,selection,hog.winSize);
    cout << "\t\tAfter resize: " << selection.cols << " x " << selection.rows <<  endl;
    //Show the training image cutted from the original image
    // imshow(category+" positive ",selection);
    // waitKey(0); 

    //Check a valid image input
    if(selection.empty()){
      featureVector.clear();
      cout << "Error: HOG image " << imageFilename.c_str()  <<  " is empty, features calculation skipped!" << endl;
      return;
    }

    //Check a valid image size
    if(selection.cols != hog.winSize.width || selection.rows != hog.winSize.height){
      featureVector.clear();
      cout << "Error: Image " << imageFilename.c_str() << " dimensions (" << selection.cols 
        << "x" << selection.rows << ") do not match HOG window size (" << hog.winSize.width 
        << "x" << hog.winSize.height << ")" << endl;
    }

    hog.compute(selection,featureVector,Size(8,8),Size(0,0));
    cout << "\t\tComputed patch HOG features" << endl;
    svm.writeFeatureVectorToFile(featureVector,false); 
    selection.release();
    featureVector.clear();
    cout << "\t\tClean exit" << endl;
  }
  imageData.release();              // we don't need the original image anymore
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
  LibSVM::SVMClassifier classifier(svmModelFile);
  cout << "Getting the Classifier" << endl;
  vector<float> descriptorVector = classifier.getDescriptorVector();
  cout << "Getting the Descriptor Vector" << endl;
  hog.setSVMDetector(descriptorVector);
  cout << "Applied the descriptor to the hog class" << endl;
  //hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
  vector<string> testImagesPos;
  vector<string> testImagesNeg;
  getSamples("/Users/david/Documents/Development/VOC2007/VOCdevkit/VOC2007/ImageSets/Main/"+category+"_val.txt",testImagesPos,testImagesNeg);
  for(int i = 0; i < testImagesPos.size(); ++i){
    Mat image = imread(testImagesPos[i],CV_LOAD_IMAGE_COLOR);
    detectTest(hog,image);
    imshow("HOG Custom Detection",image);
    waitKey(0);
  }
}

int main(int argc, char** argv){
  HOGDescriptor hog;
  hog.winSize = Size(64,128);
  category = argv[1];
  
  //Get the positive and negative training samples
  static vector<string> positiveTrainingImages;
  static vector<string> negativeTrainingImages;
  static vector<string> validExtensions;
  
  //Get the images found in the list for the category
  sampleListPath = sampleListPath+category+"_train.txt";
  getSamples(sampleListPath, positiveTrainingImages, negativeTrainingImages);

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

  LibSVM::SVMTrainer svm(featuresFile.c_str());
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
    calculateFeaturesFromInput(currentImageFile, hog, svm);
    
    //os.clear(); os.seekp(0);    // reset string stream

  }
  printf("\n");
  cout << "Finished writing the features" << endl;

  // Starting the training of the model
  cout << "Starting the training of the model using LibSVM" << endl;
  svm.trainAndSaveModel(svmModelFile,LINEAR);
  cout << "SVM Model saved to " << svmModelFile << endl;
  
  // Test the just trained descriptor
  test();

  return 0;
}
