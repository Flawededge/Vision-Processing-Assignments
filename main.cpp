#include <stdio.h>
#include <chrono>
#include <ctime>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "opencv2/ml/ml.hpp"
#include <cstdio>
#include <vector>

using namespace cv;
using namespace std;
using namespace chrono;
using namespace cv::ml;

#define Mpixel(image,x,y) image.at<cv::Vec3b>(y, x)
#define MpixelB(image,x,y) (uchar)image.at<cv::Vec3b>(y, x)[0]
#define MpixelG(image,x,y) (uchar)image.at<cv::Vec3b>(y, x)[1]
#define MpixelR(image,x,y) (uchar)image.at<cv::Vec3b>(y, x)[2]

#define setName "train.xml"

/*********************************************************************************************
 * compile with:
 * g++ -std=c++11 camera_with_fps.cpp -o camera_with_fps `pkg-config --cflags --libs opencv`
*********************************************************************************************/

vector<float> FourierDescriptor(vector<Point>& contour);
int getMaxAreaContourId(vector <vector<cv::Point>> contours);
template<typename T> static Ptr<T> load_classifier(const string& filename_to_load);


Mat frame;//, image;
int main(int argc, char** argv)
{
  /// Force load an image
	argc = 2;
	argv[1] = "C:\\Users\\Ben\\Documents\\,Git\\Vision-Processing-Assignments\\testSet\\1_A.jpg";

	Ptr<ANN_MLP> model;
	model = load_classifier<ANN_MLP>(setName);
	if (model.empty())
		cout << "Classifier empty =(";
		return 2;


  /// Load file from parameter
	if (argc == 2) {
		Mat image = imread(argv[1]);

		if (image.empty()) {
			cout << "Image empty. May have read wrong =(" << endl;
			return 1;
		}

		// Do the processing here!
	  // Resize the image to a set size for ease of viewing
		double scale = 500.0 / double(image.size[0]); // Find the % scale
		resize(image, image, Size(), scale, scale); // Scale the image

	  // Threshold the image to get just the hand
		Mat outImage;
		Mat sleeveImage;
		inRange(image, Scalar(0, 0, 120), Scalar(255, 255, 255), sleeveImage);
		inRange(image, Scalar(190, 0, 100), Scalar(255, 255, 255), image); // Threshold red, as hand is brown
		image = ~image - ~sleeveImage;
		imshow("OUT", image);
		imshow("Sleeve", sleeveImage);
		medianBlur(image, image, 11); // Median to remove large blobs

	  // Extract the contours
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours(image, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	  // Get the largest contour and display it
		Mat contourImage(image.size(), CV_8UC3); // Create a blank image for the contours to be drawn on
		contourImage = Scalar(0, 0, 0);
		int id = getMaxAreaContourId(contours);
		drawContours(contourImage, contours, id, Scalar(0, 255, 0), 1);
		imshow("Contours", contourImage);

	  // Calculate and show the descriptor
		vector<float> CE;
		CE = FourierDescriptor(contours[id]); // Get the descripton


		// ----------------------------
		
		imshow("Output image", image);
		waitKey(0);
		return(0);
	}

  /// Use webcam stream
	else {
		VideoCapture cap;
		cap.open(0);
		if (!cap.isOpened())
		{
			cout << "Failed to open camera" << endl;
			return 0;
		}
		cout << "Opened camera" << endl;
		namedWindow("WebCam", 1);
		cap.set(CAP_PROP_FRAME_WIDTH, 640);
		//   cap.set(CV_CAP_PROP_FRAME_WIDTH, 960);
		//   cap.set(CV_CAP_PROP_FRAME_WIDTH, 1600);
		cap.set(CAP_PROP_FRAME_HEIGHT, 480);
		//   cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
		//   cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1200);
		cap >> frame;
		printf("frame size %d %d \n", frame.rows, frame.cols);
		int key = 0;

		double fps = 0.0;
		while (1) {
			system_clock::time_point start = system_clock::now();
			//for(int a=0;a<10;a++){
			cap >> frame;
			if (frame.empty())
				break;

			// Process the frame here


			// ----------------------
			char printit[100];
			sprintf(printit, "%2.1f", fps);
			putText(frame, printit, Point(10, 30), FONT_HERSHEY_PLAIN, 2, Scalar(255, 255, 255), 2, 8);
			imshow("WebCam", frame);
			key = waitKey(1);
			if (key == 113 || key == 27) return 0;//either esc or 'q'

		  //}
			system_clock::time_point end = system_clock::now();
			double seconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
			//fps = 1000000*10.0/seconds;
			fps = 1000000 / seconds;
			cout << "frames " << fps << " seconds " << seconds << endl;
		}
	}
}

//if (!filename_to_load.empty())
//{
//	model = load_classifier<ANN_MLP>(filename_to_load);
//	if (model.empty())
//		return false;
//	//ntrain_samples = 0;
//}











vector<float> FourierDescriptor(vector<Point>& contour) {
	vector<float> CE;
	vector <float> ax, ay, bx, by;

	int m = contour.size();
	int n = 9;
	float t = (2 * 3.141529) / m;

	for (int k = 0; k < n; k++) {
		ax.push_back(0.0); ay.push_back(0.0);
		bx.push_back(0.0); by.push_back(0.0);

		for (int i = 0; i < m; i++) {
			ax[k] += contour[i].x * cos((k + 1) * t * (i));
			bx[k] += contour[i].x * sin((k + 1) * t * (i));
			ay[k] += contour[i].y * cos((k + 1) * t * (i));
			by[k] += contour[i].y * sin((k + 1) * t * (i));
		}
		ax[k] /= m;
		bx[k] /= m;
		ay[k] /= m;
		by[k] /= m;
	}

	for (int k = 0; k < n; k++) {
		float working1 = (ax[k] * ax[k] + ay[k] * ay[k]);
		float working2 = (ax[0] * ax[0] + ay[0] * ay[0]);
		float working3 = (bx[k] * bx[k] + by[k] * by[k]);
		float working4 = (bx[0] * bx[0] + by[0] * by[0]);

		CE.push_back(sqrt(working1 / working2) + sqrt(working3 / working4));
	}
	return CE;
}


int getMaxAreaContourId(vector <vector<cv::Point>> contours) {
	/* Get max area contour ID
	This small function returns the id of the largest contour by area
	Note: Retrieved from a guy on Stack Overflow
	*/
	double maxArea = 0;
	int maxAreaContourId = -1;
	for (int j = 0; j < contours.size(); j++) {
		double newArea = cv::contourArea(contours.at(j));
		if (newArea > maxArea) {
			maxArea = newArea;
			maxAreaContourId = j;
		}
	}
	return maxAreaContourId;
}

template<typename T> static Ptr<T> load_classifier(const string& filename_to_load)
{
	// load classifier from the specified file
	Ptr<T> model = StatModel::load<T>(filename_to_load);
	if (model.empty())
		cout << "Could not read the classifier " << filename_to_load << endl;
	else
		cout << "The classifier " << filename_to_load << " is loaded.\n";

	return model;
}