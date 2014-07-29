#include <stdio.h>
#include <fstream>
#include <dirent.h>
#include <numeric>

#include <opencv2/opencv.hpp>

#include "svm_light_wrapper.h"

using namespace std;
using namespace cv;

static string testPath = "/Users/david/Documents/Development/INRIAPerson/Test/";
static string testAnnotationsPath = "/Users/david/Documents/Development/INRIAPerson/Test/annotations/";
static string featuresFile = "/Users/david/Documents/HOGINRIATraining/genfiles/features.dat";
static string svmModelFile = "/Users/david/Documents/HOGINRIATraining/genfiles/svmModel.dat";
static string descriptorVectorFile = "/Users/david/Documents/HOGINRIATraining/genfiles/descriptorVector.dat";

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

static void getScore(Mat& imageData, const vector<Rect>& found, string& testImageFile, bool c, int& truePositive, int& falsePositive, int& numPositives){
	//Compare the detected boxes with the ground truth
	if(c==true){
		string annPath = getAnnotation(testImageFile,testAnnotationsPath);
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
					Rect groundTruth(xmin,ymin,xmax-xmin,ymax-ymin);
					numPositives++;
					//rectangle(imageData,groundTruth.tl(), groundTruth.br(), Scalar(255, 255, 255), 3);
					for(int i = 0; i < found.size(); ++i){
						Rect intersection = groundTruth & found[i];
						//double sumArea = (groundTruth.width*groundTruth.height) + (found[i].width*found[i].height);
						double sumArea = groundTruth.width*groundTruth.height;
						double intersectionArea = intersection.width * intersection.height;
						// cout << "Intersection Area = " << intersectionArea << endl;
						// cout << "sumArea = " << sumArea << endl;
						// cout << "Intersection area for found " << i << ": " << intersectionArea/sumArea << endl;
						if((intersectionArea/sumArea) > 0.5){
							truePositive++;
							break;
						}
						else
							falsePositive++;
						
						// rectangle(imageData,found[i].tl(), found[i].br(), Scalar(0, 0, 0), 3);
						// imshow("Test image",imageData);
						// waitKey(0);
					}
				}
			}
			annFile.close();
		}
		else
			cout << "Couldn't open the annontations file: " << annPath << endl;
	}
	else{
		numPositives++;
		if(found.size() > 0)
			falsePositive += found.size();
	}
} 

static void detectTest(const HOGDescriptor& hog, string& testImageFile, bool c, int& truePositive, int& falsePositive, int& numPositives){
	Mat imageData = imread(testImageFile,CV_LOAD_IMAGE_GRAYSCALE);
	//resize(imageData,imageData,Size(200,200));
	vector<Rect> found;
	double groupThreshold = 2.0;
	Size padding(Size(0,0));
	Size winStride(Size(16,16));
	double hitThreshold = 0.0;
	hog.detectMultiScale(imageData,found, hitThreshold,winStride,padding,1.01,groupThreshold);
	getScore(imageData,found, testImageFile,c, truePositive, falsePositive, numPositives);

	cout << "Number of true positives: " << truePositive << endl;
	cout << "Number of false positives: " << falsePositive << endl;
	cout << "---------------------------" << endl;
}

int main(int argc, char** argv){
	HOGDescriptor hog;
	hog.winSize = Size(64,128);
	SVMLight::SVMClassifier classifier(svmModelFile);
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

	vector<int> truePositives;
	vector<int> falsePositives;
	getSamples(testPath, testImagesPos, testImagesNeg, validExtensions);
	// for(int i = 0; i < testImagesPos.size(); ++i){
	int np = 0;
	for(int i = 0; i < testImagesPos.size(); ++i){
		cout << "Testing image: " << testImagesPos[i] << endl;
		int tp = 0;
		int fp = 0;
		int n = 0;
		detectTest(hog,testImagesPos[i],true, tp,fp,n);
		truePositives.push_back(tp);
		falsePositives.push_back(fp);
		np += n;
	}
	for(int i = 0; i < testImagesNeg.size(); ++i){
		cout << "Testing image: " << testImagesNeg[i] << endl;
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

	cout << "Tested " << truePositives.size() << " images ..." << endl;
	cout << "Total number of positives: " << np << endl;
	cout << "Total number of true positives: " << sumTruePositives << endl;
	cout << "Total number of false positives: " << sumFalsePositives << endl;
	cout << "Precision: " << (double)sumTruePositives/(sumTruePositives+sumFalsePositives) << endl;
	cout << "Recall: " << (double)sumTruePositives/np << endl;

}




