// https://github.com/Flawededge/Vision-Processing-Assignments

#include <stdio.h>
#include <chrono>
#include <ctime>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;
using namespace chrono;

char encodingarray[64] = { ' ','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','x','y','w','z',
'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','X','Y','W','Z',
'0','1','2','3','4','5','6','7','8','9','.' };

/*********************************************************************************************
 * compile with:
 * g++ -std=c++11 camera_with_fps.cpp -o camera_with_fps `pkg-config --cflags --libs opencv`
*********************************************************************************************/

Mat rotate_qt(Mat inImage);

Mat frame;//, image;
int main(int argc, char** argv)
{
	String imagepath = "C:\\Users\\crdig\\Google Drive\\Uni\\Code\\Training Networks\\Images\\abcde.jpg";

	Mat image = imread(imagepath);
	if (image.empty()) {
		cout << "Image empty" << endl;
		return 1;
	}

	Mat rotatedImage = rotate_qt(image);
	imshow("CurrentImage", rotatedImage);
	waitKey(0);

	return(0);

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

Mat rotate_qt(Mat inImage) {
	Mat workingImage;
	inRange(inImage, Scalar(200, 0, 0), Scalar(250, 10, 10), workingImage);

	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(workingImage, workingImage, kernel);

	vector<Vec3f> circles;
	HoughCircles(workingImage, circles, HOUGH_GRADIENT, 1, 1, 100, 100, 0, 50);

	vector<Vec3f> corners;
	for (size_t i = 0; i < circles.size(); i++)
	{
		Vec3i c = circles[i]; // Current circle

		Point center = Point(c[0], c[1]); // circle center
		circle(inImage, center, 1, Scalar(0, 100, 100), 3, LINE_AA); // circle outline
		int radius = c[2];
		circle(inImage, center, radius, Scalar(255, 0, 255), 3, LINE_AA);
		Vec3f curThing = circles[i];

		for (int j = 0; j < circles.size(); j++) {

			Point otherCenter = Point(circles[j][0], circles[j][1]); // circle center
			if (norm(center - otherCenter) < 20) {

			}
		}
		
	}



	cout << "Testing" << endl;

	return inImage;
}

String read_qr(Mat* inImage, Mat* outImage) {
	return "Why are you running this?";
}