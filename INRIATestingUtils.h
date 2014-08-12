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
		void testImage(string& imagePath, const HOGDescriptor& hog, vector<Rect>& found, vector<Rect>& falsePositives, vector<Rect>& truePositives, bool isPositive);
		vector<Rect> getROI(const string& imageFilename, bool c);

	private:
		string getAnnotation(const string imagePath);
		void get(vector<Rect> found, string imagePath, bool isPositive, vector<Rect>& falsePositives, vector<Rect>& truePositives);
		bool validatePath(string path);
		
	};
}