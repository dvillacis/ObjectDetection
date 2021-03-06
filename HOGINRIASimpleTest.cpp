#include <stdio.h>
#include <dirent.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include <boost/filesystem.hpp>

#include "INRIATestingUtils.h"

#include "svm_wrapper.h"

using namespace cv;
using namespace std;
using namespace boost;

static string testPath = "/Users/david/Documents/Development/INRIAPerson/Test/";
static string annPath = "/Users/david/Documents/Development/INRIAPerson/Test/annotations/";

static void getSamples(string listPath, vector<string>& filenames, const vector<string>& validExtensions){
  LOG(INFO) << "Getting files in: " << listPath;
  filesystem::path p = listPath;
  try{
    if(filesystem::exists(p) && filesystem::is_directory(p)){
      filesystem::directory_iterator end_iter;
      for(filesystem::directory_iterator dir_iter(p); dir_iter != end_iter; ++dir_iter){
        filesystem::path imPath = *dir_iter;
        if(find(validExtensions.begin(), validExtensions.end(), imPath.extension().string()) != validExtensions.end()){
          filenames.push_back(imPath.string());
        }
      }
    } 
  }
  catch (const filesystem::filesystem_error& ex){
    LOG(ERROR) << ex.what();
  }
}

int main(int argc, char** argv){

	INRIAUtils::INRIATestingUtils* utils = new INRIAUtils::INRIATestingUtils();
	utils->setAnnotationsPath(annPath);

	//Starting the logging library
	FLAGS_logtostderr = true;
	FLAGS_stderrthreshold = 0;
	google::InitGoogleLogging(argv[0]);

	HOGDescriptor hog;
	hog.winSize = Size(64,128);
	LibSVM::SVMClassifier classifier(argv[1]);
	vector<float> descriptorVector = classifier.getDescriptorVector();
	LOG(INFO) << "Getting the Descriptor Vector";
	hog.setSVMDetector(descriptorVector);
	LOG(INFO) << "Applied the descriptor to the hog class";
	//hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	vector<string> testImagesPos;
	vector<string> testImagesNeg;
	static vector<string> validExtensions;
	validExtensions.push_back(".jpg");
	validExtensions.push_back(".png");

	//Getting positive testing samples
	getSamples(testPath+"pos/", testImagesPos, validExtensions);
	//Getting negative testing samples
	getSamples(testPath+"neg/", testImagesNeg, validExtensions);

	for(int i = 0; i < 30; ++i){
		int index = rand() % testImagesPos.size()-1;
		Mat imageData = imread(testImagesPos[index],CV_LOAD_IMAGE_COLOR);
		vector<Rect> found;
		vector<Rect> groundTruth;
		utils->getDetections(testImagesPos[index], hog, found, true);
		utils->getGroundTruth(testImagesPos[index], groundTruth);
		utils->showImage(testImagesPos[index], groundTruth, found);
	}
	for(int i = 0; i < 30; ++i){
		int index = rand() % testImagesNeg.size()-1;
		Mat imageData = imread(testImagesNeg[index],CV_LOAD_IMAGE_COLOR);
		vector<Rect> found;
		vector<Rect> groundTruth;
		utils->getDetections(testImagesNeg[index], hog, found, false);
		utils->showImage(testImagesNeg[index], groundTruth, found);
	}
}