#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

using namespace cv;
using namespace std;
using namespace boost;

namespace INRIAUtils
{
	class INRIATestingUtils{
	public:
		INRIATestingUtils();
		virtual ~INRIATestingUtils();

		void setAnnotationsPath(const string annPath);
		string getAnnotationsPath();

		void getDetections(string imagePath, const HOGDescriptor& hog, vector<Rect>& found_filtered, bool isPositive);
		void getGroundTruth(string imagePath, vector<Rect>& groundTruth);
		float getOverlapArea(Rect detection, Rect groundTruth);

		vector<Rect> getROI(const string& imageFilename, bool c);
		void getSamples(string listPath, vector<string>& filenames, const vector<string>& validExtensions);
		void showImage(string imagePath, vector<Rect> groundTruth, vector<Rect> found);

	private:
		string getAnnotation(const string imagePath);
		void get(vector<Rect> found, string imagePath, bool isPositive, vector<Rect>& falsePositives, vector<Rect>& truePositives);
		bool validatePath(string path);
		
	};
}