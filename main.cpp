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

#define Mpixel(image,x,y) image.at<cv::Vec3b>(y, x)
#define MpixelB(image,x,y) (uchar)image.at<cv::Vec3b>(y, x)[0]
#define MpixelG(image,x,y) (uchar)image.at<cv::Vec3b>(y, x)[1]
#define MpixelR(image,x,y) (uchar)image.at<cv::Vec3b>(y, x)[2]

/*********************************************************************************************
 * compile with:
 * g++ -std=c++11 camera_with_fps.cpp -o camera_with_fps `pkg-config --cflags --libs opencv`
*********************************************************************************************/

Mat rotate_qr(Mat inImage);
String read_qr(Mat inImage);
int getMaxAreaContourId(vector <vector<cv::Point>> contours);

Mat frame;//, image;
int main(int argc, char** argv)
{
	//String imagepath[] = {
	//	"C:\\Users\\Ben\\Documents\\,Git\\Vision-Processing-Assignments\\Images\\abcde.jpg",
	//	"C:\\Users\\Ben\\Documents\\,Git\\Vision-Processing-Assignments\\Images\\abcde_rotated.jpg",
	//	"C:\\Users\\Ben\\Documents\\,Git\\Vision-Processing-Assignments\\Images\\abcde_rotated_scaled.jpg",
	//	"C:\\Users\\Ben\\Documents\\,Git\\Vision-Processing-Assignments\\Images\\abcde_scaled.jpg",
	//	"C:\\Users\\Ben\\Documents\\,Git\\Vision-Processing-Assignments\\Images\\congratulations.jpg",
	//	"C:\\Users\\Ben\\Documents\\,Git\\Vision-Processing-Assignments\\Images\\congratulations_rotated.jpg",
	//	"C:\\Users\\Ben\\Documents\\,Git\\Vision-Processing-Assignments\\Images\\congratulations_rotated_scaled.jpg",
	//	"C:\\Users\\Ben\\Documents\\,Git\\Vision-Processing-Assignments\\Images\\congratulations_scaled.jpg",
	//	"C:\\Users\\Ben\\Documents\\,Git\\Vision-Processing-Assignments\\Images\\Darwin.jpg",
	//	"C:\\Users\\Ben\\Documents\\,Git\\Vision-Processing-Assignments\\Images\\Darwin_rotated.jpg",
	//	"C:\\Users\\Ben\\Documents\\,Git\\Vision-Processing-Assignments\\Images\\Darwin_rotated_scaled.jpg",
	//	"C:\\Users\\Ben\\Documents\\,Git\\Vision-Processing-Assignments\\Images\\Darwin_scaled.jpg",
	//	"C:\\Users\\Ben\\Documents\\,Git\\Vision-Processing-Assignments\\Images\\farfaraway.jpg",
	//	"C:\\Users\\Ben\\Documents\\,Git\\Vision-Processing-Assignments\\Images\\farfaraway_rotated.jpg",
	//	"C:\\Users\\Ben\\Documents\\,Git\\Vision-Processing-Assignments\\Images\\farfaraway_rotated_scaled.jpg",
	//	"C:\\Users\\Ben\\Documents\\,Git\\Vision-Processing-Assignments\\Images\\farfaraway_scaled.jpg" };

	//String savePath[] = {
	//	"C:\\Users\\Ben\\Documents\\,Git\\Vision-Processing-Assignments\\ImageOut\\abcde.jpg",
	//	"C:\\Users\\Ben\\Documents\\,Git\\Vision-Processing-Assignments\\ImageOut\\abcde_rotated.jpg",
	//	"C:\\Users\\Ben\\Documents\\,Git\\Vision-Processing-Assignments\\ImageOut\\abcde_rotated_scaled.jpg",
	//	"C:\\Users\\Ben\\Documents\\,Git\\Vision-Processing-Assignments\\ImageOut\\abcde_scaled.jpg",
	//	"C:\\Users\\Ben\\Documents\\,Git\\Vision-Processing-Assignments\\ImageOut\\congratulations.jpg",
	//	"C:\\Users\\Ben\\Documents\\,Git\\Vision-Processing-Assignments\\ImageOut\\congratulations_rotated.jpg",
	//	"C:\\Users\\Ben\\Documents\\,Git\\Vision-Processing-Assignments\\ImageOut\\congratulations_rotated_scaled.jpg",
	//	"C:\\Users\\Ben\\Documents\\,Git\\Vision-Processing-Assignments\\ImageOut\\congratulations_scaled.jpg",
	//	"C:\\Users\\Ben\\Documents\\,Git\\Vision-Processing-Assignments\\ImageOut\\Darwin.jpg",
	//	"C:\\Users\\Ben\\Documents\\,Git\\Vision-Processing-Assignments\\ImageOut\\Darwin_rotated.jpg",
	//	"C:\\Users\\Ben\\Documents\\,Git\\Vision-Processing-Assignments\\ImageOut\\Darwin_rotated_scaled.jpg",
	//	"C:\\Users\\Ben\\Documents\\,Git\\Vision-Processing-Assignments\\ImageOut\\Darwin_scaled.jpg",
	//	"C:\\Users\\Ben\\Documents\\,Git\\Vision-Processing-Assignments\\ImageOut\\farfaraway.jpg",
	//	"C:\\Users\\Ben\\Documents\\,Git\\Vision-Processing-Assignments\\ImageOut\\farfaraway_rotated.jpg",
	//	"C:\\Users\\Ben\\Documents\\,Git\\Vision-Processing-Assignments\\ImageOut\\farfaraway_rotated_scaled.jpg",
	//	"C:\\Users\\Ben\\Documents\\,Git\\Vision-Processing-Assignments\\ImageOut\\farfaraway_scaled.jpg" };

	
	Mat image = imread("C:\\Users\\crdig\\Documents\\,Git\\Vision Processing Assignments\\Images\\congratulations_rotated_scaled.jpg");
	if (image.empty()) {
		cout << "Image empty =(" << endl;
		return 1;
	}

	Mat rotatedImage = rotate_qr(image);
	String readData = read_qr(rotatedImage);

	cout << "Read data: " << readData << endl;
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

Mat rotate_qr(Mat inImage) {
	Mat workingImage;
	resize(inImage, workingImage, Size(640, 640));
	cvtColor(workingImage, workingImage, COLOR_BGR2HSV);

	inRange(workingImage, Scalar(100, 0, 0), Scalar(140, 255, 255), workingImage);

	// Find the outside bounds of the QR code
	Mat kernel = getStructuringElement(MORPH_RECT, Size(20, 20)); // Lagre kernel to pick up noise
	dilate(workingImage, workingImage, kernel, Point(-1, -1), 1); // Make sure it's solid

		// Find contours
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(workingImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	int largest = getMaxAreaContourId(contours);  // The largest contour will be the QR code

	RotatedRect outline = minAreaRect(contours[largest]);
	Mat M = getRotationMatrix2D(Point(inImage.cols/2, inImage.rows/2), outline.angle, 1);
	warpAffine(inImage, inImage, M, inImage.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 255, 255));

	// The QR code is squared up by this point, time to crop the square

	cvtColor(inImage, workingImage, COLOR_BGR2HSV);
	inRange(workingImage, Scalar(0, 0, 0), Scalar(180, 255, 240), workingImage);

	// Find the outside bounds of the QR code
	dilate(workingImage, workingImage, kernel, Point(-1, -1), 1); // Make sure it's solid
	findContours(workingImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	
	largest = getMaxAreaContourId(contours);  // The largest contour will be the QR code

	// Get the ROI rectangle
	Rect newSize = boundingRect(Mat(contours[largest]));
	inImage = inImage(newSize); // Crop image
	resize(inImage, inImage, Size(640, 640));

	// Time to work out the final rotation of the image
	cvtColor(inImage, workingImage, COLOR_BGR2GRAY);

	contours.clear();
	findContours(~workingImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	largest = getMaxAreaContourId(contours);  // The largest contour will be the QR code
	//drawContours(inImage, contours, largest, Scalar(255, 0, 0), 1);

	Rect FinalQR = boundingRect(Mat(contours[largest])); // Get the bounding for the current Qr

	// Create a rectangle in each corner
	Rect topLeft(FinalQR.x, FinalQR.y, 70, 70);
	Rect topRight(FinalQR.x + FinalQR.width - 70, FinalQR.y, 70, 70);
	Rect botLeft(FinalQR.x, FinalQR.y + FinalQR.height - 70, 70, 70);
	Rect botRight(FinalQR.x + FinalQR.width - 70, FinalQR.y + FinalQR.height - 70, 70, 70);

	Scalar means[] = { mean(inImage(topRight)), mean(inImage(topLeft)), mean(inImage(botLeft)), mean(inImage(botRight)) };

	//rectangle(inImage, topLeft, Scalar(0, 255, 0), 2);
	//rectangle(inImage, topRight, Scalar(0, 255, 0), 2);
	//rectangle(inImage, botLeft, Scalar(0, 255, 0), 2);
	//rectangle(inImage, botRight, Scalar(0, 255, 0), 2);

	// Compare the mean pixel values of each corner. The one with the most entropy will be the non-circle corner
	int entropy[4] = { 0, 0, 0, 0 };
	for (int i = 0; i < 3; i++) {
		int med = (means[0][i] + means[1][i] + means[2][i] + means[3][i]) / 4; // Get the median value of the current channel
		for (int j = 0; j < 4; j++) {
			entropy[j] += abs(means[j][i] - med); // Get the entropy of the current channel and add it to the total
		}
	}
	const int N = sizeof(entropy) / sizeof(int);
	int rotationAngle = distance(entropy, max_element(entropy, entropy + N)) * 90;
	//cout << "Rotation: " << rotationAngle << endl;
	M = getRotationMatrix2D(Point(inImage.cols / 2, inImage.rows / 2), -rotationAngle, 1);
	warpAffine(inImage, inImage, M, inImage.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 255, 255));

	return inImage;
}



String read_qr(Mat inImage) {
/* Read QR
- Takes in an image containing a cropped color qr code and reads the data from it

1. Get bounding rectangle
2. Loop through each square, missing the corners to read the pixel values and convert them to data
	2.1. During this use each 6 bits of data to get a character from the encodingarray
*/
	const float gridSize = 47; // The X*X size of the grid
	const int cornerSize = 6; // The X*X corner size of the grid
	

	/// Get an accurate bounding rectangle
	Mat workingImage;
	cvtColor(inImage, workingImage, COLOR_BGR2GRAY);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	inRange(~workingImage, 100, 255, workingImage);
	findContours(workingImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	Rect qr = boundingRect(Mat(contours[getMaxAreaContourId(contours)]));  // The largest contour will be the QR code

	double step = qr.width / gridSize;
	double offset = 7 + double(qr.x);
	int farThresh = gridSize - cornerSize;

	rectangle(inImage, qr, Scalar(0, 255, 255));

	/// Start of processing the squares of the image
	Point cur;
	

	const char encodingarray[64] = { ' ','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','x','y','w','z',
'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','X','Y','W','Z',
'0','1','2','3','4','5','6','7','8','9','.' };

	Vec3b prevValue;
	String output;
	bool first = true;

	for (int y = 0; y < gridSize; y++) {
		cur.y = round(y * step + offset); // Get the Y coordiante
		for (int x = 0; x < gridSize; x++) {
			if (((x < cornerSize || x >= farThresh) && y >= farThresh) || 
				(x < cornerSize && y < cornerSize)) continue; // Check if in the corner

			cur.x = round(x * step + offset); // Get the X coordiante

			//cout << x << " " << y << " " << Mpixel(inImage, cur.x, cur.y) << endl;
			
			if (first) {
				prevValue = Mpixel(inImage, cur.x, cur.y); // Get the pixel data at the current coordinate
			}
			else {
				int value = 0;
				value += prevValue[2] > 128 ? 32 : 0;
				value += prevValue[1] > 128 ? 16 : 0;
				value += prevValue[0] > 128 ? 8 : 0;
				value += Mpixel(inImage, cur.x, cur.y)[2] > 128 ? 4 : 0;
				value += Mpixel(inImage, cur.x, cur.y)[1] > 128 ? 2 : 0;
				value += Mpixel(inImage, cur.x, cur.y)[0] > 128 ? 1 : 0;
				output += encodingarray[value];

				// Output for debug
				//cout << value << "|" << output << "|" << endl; 
				//imshow("WorkingImage", inImage);
				//waitKey(0);
			}
			first = !first;

			circle(inImage, cur, 2, Scalar(0, 0, 255)); // Draw circles to make it look like the image has chicken pox
		}
	}

	
	imshow("Input", inImage); // Show the image just because it's interesting
	return output;
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