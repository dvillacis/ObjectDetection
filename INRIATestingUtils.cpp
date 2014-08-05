#include <glog/logging.h>
#include <fstream>

#include "INRIATestingUtils.h"

using namespace std;
using namespace cv;

namespace INRIAUtils{
	string annotationsPath;
	//Constructor
	INRIATestingUtils::INRIATestingUtils(string annPath){
		annotationsPath = annPath;
	}

	//Destructor
	INRIATestingUtils::~INRIATestingUtils(){

	}

	void INRIATestingUtils::testImage(string& imagePath, const HOGDescriptor& hog, vector<Rect>& falsePositives, vector<Rect>& truePositives, bool isPositive){
		LOG(INFO) << "Testing image: " << imagePath;

		Mat image = imread(imagePath, CV_LOAD_IMAGE_GRAYSCALE);
		vector<Rect> found;
		double groupThreshold = 2.0;
		Size padding(Size(0,0));
		Size winStride(Size(16,16));
		double hitThreshold = 0.0;

		hog.detectMultiScale(image,found,hitThreshold,winStride,padding,1.01,groupThreshold);

		get(found, imagePath, isPositive, falsePositives, truePositives);
	}

	vector<string> INRIATestingUtils::split(const string& s, const string& delim, const bool keep_empty) {
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

	void INRIATestingUtils::get(vector<Rect> found, string imagePath, bool isPositive, vector<Rect>& falsePositives, vector<Rect>& truePositives){
		if(isPositive == true){
			vector<Rect> falsePositives;
			string annPath = getAnnotation(imagePath);
			fstream annFile;
			annFile.open(annPath.c_str(), ios::in);
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
						for(int i = 0; i < found.size(); ++i){
							Rect intersection = groundTruth & found[i];
							double sumArea = groundTruth.width*groundTruth.height;
							double intersectionArea = intersection.width * intersection.height;
							if((intersectionArea/sumArea) > 0.5){
								truePositives.push_back(found[i]);
							}
							else
								falsePositives.push_back(found[i]);
						}
					}
				}
				annFile.close();
			}
			else
				LOG(ERROR) << "Couldn't open the annotations file: " << annPath;
		}
		else{
			for(int i = 0; i < found.size(); ++i)
				falsePositives.push_back(found[i]);	
		}
	}

	string INRIATestingUtils::getAnnotation(const string imagePath){
		//Get the annotation filename
		vector<string> parts = split(imagePath,"/");
		string imageName = split(parts[parts.size()-1],".")[0];
		string annPath = annotationsPath+imageName+".txt";
		return annPath;
	}

}