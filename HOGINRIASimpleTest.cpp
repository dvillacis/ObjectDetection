#include <stdio.h>
#include <dirent.h>
#include <opencv2/opencv.hpp>

#include "lib_svm_wrapper.h"

using namespace cv;
using namespace std;

static string testPath = "/Users/david/Documents/Development/INRIAPerson/Test/";

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

int main(int argc, char** argv){
	HOGDescriptor hog;
	hog.winSize = Size(64,128);
	LibSVM::SVMClassifier classifier(argv[1]);
	vector<float> descriptorVector = classifier.getDescriptorVector();
	cout << "Getting the Descriptor Vector" << endl;
	hog.setSVMDetector(descriptorVector);
	cout << "Applied the descriptor to the hog class" << endl;

	vector<string> testImagesPos;
	vector<string> testImagesNeg;
	static vector<string> validExtensions;
	validExtensions.push_back("jpg");
	validExtensions.push_back("png");

	getSamples(testPath, testImagesPos, testImagesNeg, validExtensions);

	for(int i = 0; i < 10; ++i){
		int index = rand() % testImagesPos.size()-1;
		cout << "Testing image: " << testImagesPos[index] << endl;
		Mat imageData = imread(testImagesPos[index],CV_LOAD_IMAGE_GRAYSCALE);
		vector<Rect> found;
		double groupThreshold = 2.0;
		Size padding(Size(0,0));
		Size winStride(Size(32,32));
		double hitThreshold = 0.2;
		hog.detectMultiScale(imageData,found, hitThreshold,winStride,padding,1.01,groupThreshold);
		for(int j = 0; j < found.size(); j++){
			rectangle(imageData,found[j].tl(),found[j].br(),Scalar(255,255,255),3);
		}
		imshow("Custom Detection",imageData);
		waitKey(0);
	}
	for(int i = 0; i < 10; ++i){
		int index = rand() % testImagesNeg.size()-1;
		cout << "Testing image: " << testImagesNeg[index] << endl;
		Mat imageData = imread(testImagesNeg[index],CV_LOAD_IMAGE_GRAYSCALE);
		vector<Rect> found;
		double groupThreshold = 2.0;
		Size padding(Size(0,0));
		Size winStride(Size(32,32));
		double hitThreshold = 0.0;
		hog.detectMultiScale(imageData,found, hitThreshold,winStride,padding,1.01,groupThreshold);
		for(int j = 0; j < found.size(); j++){
			rectangle(imageData,found[j].tl(),found[j].br(),Scalar(255,255,255),3);
		}
		imshow("Custom Detection",imageData);
		waitKey(0);
	}
}