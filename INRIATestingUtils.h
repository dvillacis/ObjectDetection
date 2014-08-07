#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

namespace INRIAUtils
{
	class INRIATestingUtils{
	public:
		INRIATestingUtils(string annPath);
		virtual ~INRIATestingUtils();

		void testImage(string& imagePath, const HOGDescriptor& hog, vector<Rect>& falsePositives, vector<Rect>& truePositives, bool isPositive);

	private:

		void get(vector<Rect> found, string imagePath, bool isPositive, vector<Rect>& falsePositives, vector<Rect>& truePositives);
		string getAnnotation(const string imagePath);
		vector<string> split(const string& s, const string& delim, const bool keep_empty = true);
	};
}