#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fstream>
#include <dirent.h>

using namespace cv;
using namespace std;

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
        //cout << "Found matching data file: " << ep->d_name << endl;
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
  string filename = argv[1];
  string posSamplesDir = argv[2];
  string negSamplesDir = argv[3];
  vector<float> descriptorVector;
  getDescriptorVectorFromFile(filename,descriptorVector);
  cout << "Detecting images in directories " << posSamplesDir << " and " << negSamplesDir << " using " << filename << endl;

  HOGDescriptor hog;
  hog.winSize = Size(64,128);
  hog.setSVMDetector(descriptorVector);
  //hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

  //Finding all the test images
  static vector<string> positiveTestingImages;
  static vector<string> negativeTestingImages;
  static vector<string> validExtensions;
  
  //Set the valid extensions to know image formats
  validExtensions.push_back("jpg");
  validExtensions.push_back("png");

  //Get the images found in the directories
  getFilesInDirectory(posSamplesDir,positiveTestingImages,validExtensions);
  getFilesInDirectory(negSamplesDir,negativeTestingImages,validExtensions);
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
