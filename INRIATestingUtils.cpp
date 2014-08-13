#include <glog/logging.h>
#include <fstream>

#include <boost/algorithm/string.hpp>
#include "INRIATestingUtils.h"

using namespace std;
using namespace cv;
using namespace boost;

namespace INRIAUtils{
	string annotationsPath;
	//Constructor
	INRIATestingUtils::INRIATestingUtils(){
		annotationsPath = "";
	}

	//Destructor
	INRIATestingUtils::~INRIATestingUtils(){

	}

	void INRIATestingUtils::setAnnotationsPath(const string annPath){
		annotationsPath = annPath;
	}

	string INRIATestingUtils::getAnnotationsPath(){
		return annotationsPath;
	}

	void INRIATestingUtils::getDetections(string imagePath, const HOGDescriptor& hog, vector<Rect>& found_filtered, bool isPositive){
		LOG(INFO) << "Obtaining detections for: " << imagePath;
		Mat image = imread(imagePath, CV_LOAD_IMAGE_COLOR);
		double groupThreshold = 2.0;
		Size padding(Size(32,32));
		Size winStride(Size(8,8));
		double hitThreshold = 0.0;

		vector<Rect> found;

		hog.detectMultiScale(image,found,hitThreshold,winStride,padding,1.05,groupThreshold);

		size_t i, j;
        for( i = 0; i < found.size(); i++ )
        {
            Rect r = found[i];
            for( j = 0; j < found.size(); j++ )
                if( j != i && (r & found[j]) == r)
                    break;
            if( j == found.size() )
                found_filtered.push_back(r);
        }
	}

	void INRIATestingUtils::getGroundTruth(string imagePath, vector<Rect>& groundTruth){
		LOG(INFO) << "Obtaining ground truth for: " << imagePath;
		string annPath = getAnnotation(imagePath);
		fstream annFile;
		annFile.open(annPath.c_str(), ios::in);
		if(annFile.good() && annFile.is_open()){
			string line;
			while(getline(annFile,line)){
				if(line.find("Bounding box for object") != string::npos){
					vector<string> temp;
					split(temp,line,is_any_of(" :-(),"),token_compress_on);
					int xmin = atoi(temp[10].c_str());
					int ymin = atoi(temp[11].c_str());
					int xmax = atoi(temp[12].c_str());
					int ymax = atoi(temp[13].c_str());
					Rect gt(xmin,ymin,xmax-xmin,ymax-ymin);
					groundTruth.push_back(gt);
				}
			}
			annFile.close();
		}
		else
			LOG(ERROR) << "Couldn't open the annotations file: " << annPath;
	}

	// void INRIATestingUtils::classifyDetections(vector<Rect> found, vector<Rect> groundTruth, vector<Rect> falsePositives, vector<Rect> truePositives){

	// }

	float INRIATestingUtils::getOverlapArea(Rect detection, Rect groundTruth){
		Rect intersection = detection & groundTruth;
		float intersectionArea = intersection.area();
		if(intersectionArea > 0.0){
			float unionArea = detection.area() + groundTruth.area() - intersectionArea;
			return (intersectionArea/unionArea)*100;
		}
		else
			return 0.0;
	}


	void INRIATestingUtils::getSamples(string listPath, vector<string>& filenames, const vector<string>& validExtensions){
		LOG(INFO) << "Getting files in: " << listPath;
		filesystem::path p = listPath;
		try{
			if(filesystem::exists(p) && filesystem::is_directory(p)){
				filesystem::directory_iterator end_iter;
				int i = 0;
				for(filesystem::directory_iterator dir_iter(p); dir_iter != end_iter; ++dir_iter){
					filesystem::path imPath = *dir_iter;
					if(find(validExtensions.begin(), validExtensions.end(), imPath.extension().string()) != validExtensions.end() && i < 5){
						filenames.push_back(imPath.string());
						//i++;
					}
				}
			} 
		}
		catch (const filesystem::filesystem_error& ex){
			LOG(ERROR) << ex.what();
		}
	}

	vector<Rect> INRIATestingUtils::getROI(const string& imagePath, bool c){
		vector<Rect> bboxes;
		if(c == true){
			string annPath = getAnnotation(imagePath);
			fstream annFile;
			annFile.open(annPath.c_str(),ios::in);
			if(annFile.good() && annFile.is_open()){
				string line;
				while(getline(annFile,line)){
					if(line.find("Bounding box for object") != string::npos){
						vector<string> temp;
						split(temp,line,is_any_of(" :-(),"),token_compress_on);
						int xmin = atoi(temp[10].c_str());
						int ymin = atoi(temp[11].c_str());
						int xmax = atoi(temp[12].c_str());
						int ymax = atoi(temp[13].c_str());
						bboxes.push_back(Rect(xmin,ymin,xmax-xmin,ymax-ymin));
					}
				}
				annFile.close();
			}
			else
			LOG(ERROR) << "Couldnt open annotations file for: " << annPath << endl;
		}
		else
		{
			Mat im = imread(imagePath,CV_LOAD_IMAGE_COLOR);
			//Generate 10 random images
			for(int i = 0; i < 10; ++i){
				int x = rand() % (im.cols-64);
				int y = rand() % (im.rows-128);
				Rect r(x,y,64,128);
				bboxes.push_back(r);
			}
		}

		return bboxes;
	}

	void INRIATestingUtils::showImage(string imagePath, vector<Rect> groundTruth, vector<Rect> found){
		Mat image = imread(imagePath,CV_LOAD_IMAGE_COLOR);
		for(int i = 0; i < groundTruth.size(); ++i)
			rectangle(image,groundTruth[i].tl(),groundTruth[i].br(),Scalar(0,0,255),2);
		for(int j = 0; j < found.size(); ++j)
			rectangle(image,found[j].tl(),found[j].br(),Scalar(255,0,0),2);
		imshow("Custom Detection",image);
		waitKey(0);
	}

	string INRIATestingUtils::getAnnotation(const string imagePath){
		//Get the annotation filename
		vector<string> parts;
		boost::split(parts,imagePath,is_any_of("/."),token_compress_on);
		string imageName = parts[parts.size()-2];
		string annPath = annotationsPath+imageName+".txt";
		return annPath;
	}

	bool INRIATestingUtils::validatePath(string path){
		filesystem::path p(path);
		try{
			if(filesystem::exists(p) && filesystem::is_regular_file(p))
				return true;
			return false;
		}
		catch (const filesystem::filesystem_error& ex){
			LOG(ERROR) << ex.what();
			return false;
		}
	}

}









