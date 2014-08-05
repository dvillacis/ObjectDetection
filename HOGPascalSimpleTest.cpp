#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <glog/logging.h>
#include "svm_wrapper.h"

#include <fstream>
#include <dirent.h>

using namespace cv;
using namespace std;

static string sampleImagesPath = "/Users/david/Documents/Development/VOC2007/VOCdevkit/VOC2007/JPEGImages/";

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
      if(atof(tokens[1].c_str()) < 0)
        posFilenames.push_back(sampleImagesPath+tokens[0]+".jpg");
      else
        negFilenames.push_back(sampleImagesPath+tokens[0]+".jpg");
    }
    File.close();
  }
}

static void getDescriptorVectorFromFile(string& filename, vector<float>& descriptorVector){
  cout << "Opening descriptor file " << filename.c_str() << endl;
  string separator = " ";
  fstream File;
  File.open(filename.c_str(), ios::in);
  if(File.good() && File.is_open()){
    string line;
    while(getline(File,line)){
      size_t pos = 0;
      string token;
      while((pos = line.find(separator)) != std::string::npos){
      	token = line.substr(0,pos);
      	descriptorVector.push_back(atof(token.c_str()));
      	line.erase(0,pos+separator.length());
      }
    }
    File.close();
  }
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

int main(int argc, char**argv){
  //Obtaining the descriptor
  string descriptorFileName = argv[1];
  string imagesListPath = argv[2];
  cout << "Detecting random images in " << imagesListPath << " using " << descriptorFileName << endl;

  HOGDescriptor hog;
  hog.winSize = Size(64,128);
  LibSVM::SVMClassifier classifier(descriptorFileName.c_str());
  vector<float> descriptorVector = classifier.getDescriptorVector();
  LOG(INFO) << "Getting the Descriptor Vector";
  hog.setSVMDetector(descriptorVector);
  LOG(INFO) << "Applied the descriptor to the hog class";
  //hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

  //Finding all the test images
  static vector<string> positiveTestingImages;
  static vector<string> negativeTestingImages;

  //Get the images found in the directories
  getSamples(imagesListPath,positiveTestingImages,negativeTestingImages);
  unsigned long overallSamples = positiveTestingImages.size()+negativeTestingImages.size();
  if(overallSamples == 0){
    cout << "No testing samples found, exiting..." << endl;
    return -1;
  }

  static vector<string> testingImages;
  testingImages.insert(testingImages.end(),positiveTestingImages.begin(),positiveTestingImages.end());
  testingImages.insert(testingImages.end(),negativeTestingImages.begin(),negativeTestingImages.end());
  random_shuffle(testingImages.begin(),testingImages.end());

  for(unsigned long sample = 0; sample < testingImages.size(); ++sample){
    cout << "\nTesting image:\t " << testingImages[sample] << endl; 
    Mat testImage;
    testImage = imread(testingImages[sample]);
    if(!testImage.data)
      printf("Couldnt open file\n");
    detectTest(hog,testImage);
    namedWindow("HOG Custom Detection", WINDOW_AUTOSIZE);
    imshow("HOG Custom Detection",testImage);
    int c = waitKey(0) & 255;
    if(c == 'q' || c == 'Q')
      break;
  }
  return 0;
}
