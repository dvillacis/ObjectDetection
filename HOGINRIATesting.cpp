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

static INRIAUtils::INRIATestingUtils* utils = new INRIAUtils::INRIATestingUtils();

int main(int argc, char** argv){
	
	//Starting the logging library
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
	validExtensions.push_back(".jpg");
	validExtensions.push_back(".png");

	//Get the positive images in the training folders
	utils->setAnnotationsPath(testAnnotationsPath);
	utils->getSamples(testPath+"pos/", testImagesPos, validExtensions);
	//Get the negavive images in the training folders
	utils->getSamples(testPath+"neg/", testImagesNeg, validExtensions);

	int num_detections = 0;
	for(int i = 0; i < testImagesPos.size(); ++i){
		vector<Rect> found;
		utils->getDetections(testImagesPos[i], hog, found, true);
		num_detections += found.size();
	}
	for(int i = 0; i < testImagesNeg.size(); ++i){
		vector<Rect> found;
		utils->getDetections(testImagesNeg[i], hog, found, false);
		num_detections += found.size();
	}

	Mat truePositives = Mat::zeros(num_detections,1,CV_32FC1);
	Mat falsePositives = Mat::zeros(num_detections,1,CV_32FC1);

	int index_detection = 0;

	for(int i = 0; i < testImagesPos.size(); ++i){
		vector<Rect> found;
		vector<Rect> groundTruth;
		utils->getDetections(testImagesPos[i], hog, found, true);
		utils->getGroundTruth(testImagesPos[i],groundTruth);

		// Iterate over every found detection
		for(int j = 0; j < found.size(); ++j){
			vector<Rect> gtFound;
			bool detectionFound = false;
			for(int k = 0; k < groundTruth.size(); ++k){
				if(utils->getOverlapArea(found[j],groundTruth[k]) > 50.0){
					if(find(gtFound.begin(),gtFound.end(),groundTruth[k]) == gtFound.end()){
						//cout << utils->getOverlapArea(found[j],groundTruth[k]) << endl;
						truePositives.at<float>(index_detection,1) = 1;
						falsePositives.at<float>(index_detection,1) = 0;
						gtFound.push_back(groundTruth[k]);
						cout << "True Positive: " << j << " - " << k << " --> " << utils->getOverlapArea(found[j],groundTruth[k]) << endl;
						detectionFound = true;
					}
					else{
						if(detectionFound == false){
							falsePositives.at<float>(index_detection,1) = 1;
							truePositives.at<float>(index_detection,1) = 0;
						}
					}
				}
				else{
					if(detectionFound == false){
						falsePositives.at<float>(index_detection,1) = 1;
						truePositives.at<float>(index_detection,1) = 0;
					}
				}
			}
			index_detection++;
		}

	}

	for(int i = 0; i < testImagesNeg.size(); ++i){
		vector<Rect> found;
		vector<Rect> groundTruth;
		utils->getDetections(testImagesNeg[i], hog, found, true);
		for(int j = 0; j < found.size(); ++j){
			falsePositives.at<float>(index_detection,1) = 1;
			truePositives.at<float>(index_detection,1) = 0;
			index_detection++;
		}
	}

	// Getting the cumulative sum
	for(int i = 1; i < truePositives.rows; ++i){
		truePositives.at<float>(i,1) += truePositives.at<float>(i-1,1);
	}
	for(int i = 1; i < falsePositives.rows; ++i){
		falsePositives.at<float>(i,1) += falsePositives.at<float>(i-1,1);
	}

	Mat precision = Mat::zeros(truePositives.rows,1,CV_32FC1);
	Mat recall = Mat::zeros(truePositives.rows,1,CV_32FC1);

	for(int i = 0; i < recall.rows; ++i){
		recall.at<float>(i,1) = truePositives.at<float>(i,1)/num_detections;
	}
	for(int i = 0; i < precision.rows; ++i){
		precision.at<float>(i,1) = truePositives.at<float>(i,1)/(truePositives.at<float>(i,1)+falsePositives.at<float>(i,1));
	}


	// Writing the detection file
	fstream outfile("output.txt",ios::out);
	if(!outfile.is_open()){
		LOG(ERROR) << "Couldn't write the file";
		return -1;
	}
	outfile << "PRECISION\t\tRECALL" << endl;
	for(int i = 0; i < truePositives.rows; ++i)
		outfile << precision.at<float>(i,1) << "\t\t" << recall.at<float>(i,1) << endl;

	outfile.close();

}