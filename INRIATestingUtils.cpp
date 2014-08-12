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

	void INRIATestingUtils::testImage(string& imagePath, const HOGDescriptor& hog, vector<Rect>& found_filtered, vector<Rect>& falsePositives, vector<Rect>& truePositives, bool isPositive){
		LOG(INFO) << "Testing image: " << imagePath;

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

		get(found_filtered, imagePath, isPositive, falsePositives, truePositives);

		LOG(INFO) << "Found " << found_filtered.size() << ": " << falsePositives.size() << " false positives and " << truePositives.size() << " true positives";
	}

	void INRIATestingUtils::get(vector<Rect> found, string imagePath, bool isPositive, vector<Rect>& falsePositives, vector<Rect>& truePositives){
		if(isPositive == true){
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
						Rect groundTruth(xmin,ymin,xmax-xmin,ymax-ymin);
						truePositives.push_back(groundTruth);
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









