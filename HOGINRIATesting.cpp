#include <stdio.h>
#include <fstream>
#include <dirent.h>
#include <numeric>
#include <glog/logging.h>

#include <opencv2/opencv.hpp>

#include "svm_wrapper.h"
#include "INRIATestingUtils.h"

using namespace std;
using namespace cv;

static string testPath = "/Users/david/Documents/Development/INRIAPerson/Test/";
static string testAnnotationsPath = "/Users/david/Documents/Development/INRIAPerson/Test/annotations/";

static void getSamples(string& listPath, vector<string>& posFilenames, vector<string>& negFilenames, const vector<string>& validExtensions){
	string posPath = listPath+"pos/";
	string negPath = listPath+"neg/";
	LOG(INFO) << "Getting positive files in: " << posPath;
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
	LOG(INFO) << "Getting negative files in: " << negPath;
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

static void detectTest(const HOGDescriptor& hog, string& testImageFile, bool c, int& truePositive, int& falsePositive, int& numPositives){
	
	INRIAUtils::INRIATestingUtils* utils = new INRIAUtils::INRIATestingUtils();
	utils->setAnnotationsPath(testAnnotationsPath);

	vector<Rect> found;
	vector<Rect> falsePositives;
	vector<Rect> truePositives;
	utils->testImage(testImageFile, hog, found, falsePositives, truePositives,c);

	LOG(INFO) << "Number of true positives: " << truePositives.size();
	LOG(INFO) << "Number of false positives: " << falsePositives.size();
}

int main(int argc, char** argv){
	//Starting the logging library
	//FLAGS_log_dir = logDir;
	FLAGS_logtostderr = false;
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
	validExtensions.push_back("jpg");
	validExtensions.push_back("png");

	vector<int> truePositives;
	vector<int> falsePositives;
	getSamples(testPath, testImagesPos, testImagesNeg, validExtensions);
	// for(int i = 0; i < testImagesPos.size(); ++i){
	int np = 0;
	for(int i = 0; i < testImagesPos.size(); ++i){
		LOG(INFO) << "Testing image: " << testImagesPos[i];
		int tp = 0;
		int fp = 0;
		int n = 1;
		detectTest(hog,testImagesPos[i],true, tp,fp,n);
		truePositives.push_back(tp);
		falsePositives.push_back(fp);
		np += n;
	}
	for(int i = 0; i < testImagesNeg.size(); ++i){
		LOG(INFO) << "Testing image: " << testImagesNeg[i];
		int tp = 0;
		int fp = 0;
		int n = 0;
		detectTest(hog,testImagesNeg[i],false, tp,fp,n);
		truePositives.push_back(tp);
		falsePositives.push_back(fp);
		np += n;
	}

	int sumTruePositives = std::accumulate(truePositives.begin(),truePositives.end(),0);
	int sumFalsePositives = std::accumulate(falsePositives.begin(),falsePositives.end(),0);

	LOG(INFO) << "Tested " << truePositives.size() << " images ...";
	LOG(INFO) << "Total number of positives: " << np;
	LOG(INFO) << "Total number of true positives: " << sumTruePositives;
	LOG(INFO) << "Total number of false positives: " << sumFalsePositives;
	LOG(INFO) << "Precision: " << (double)sumTruePositives/(sumTruePositives+sumFalsePositives);
	LOG(INFO) << "Recall: " << (double)sumTruePositives/np;

}