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
	if (argc == 2) {
		Mat image = imread(argv[1]);

		if (image.empty()) {
			cout << "Image empty. May have read wrong =(" << endl;
			return 1;
		}

		Mat rotatedImage = rotate_qr(image);
		String readData = read_qr(rotatedImage);
		cout << "Read data: " << readData << endl;
		waitKey(0);
		return(0);
	}
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

			frame = rotate_qr(frame);

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

Mat rotate_qr(Mat inImage) {
/* Rotate QR
- Takes in a color QR code on a very bright background and rotates it to the correct orientation

1. Make the working image small to make the process fast
2. Find the QR code by thresholding an inverted image
3. Square up the working image
4. Detect which corner is filled, and so doesn't have circles
5. Orient the original image and return it 
*/
	/// Find where the QR code is
	Mat workingImage;
	resize(inImage, workingImage, Size(300, 300)); // Make image small to speed up processing
	cvtColor(workingImage, workingImage, COLOR_BGR2GRAY); // Convert image to grayscale, as color isn't required
	inRange(~workingImage, 100, 255, workingImage); // Threshold the negative of the image to remove white background

	/// Morph the image to make it a bit more solid
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3)); // Small kernel for a small image
	erode(workingImage, workingImage, kernel, Point(-1, -1), 1); // Small morph to wear away the corners
	morphologyEx(workingImage, workingImage, MORPH_CLOSE, kernel, Point(-1,-1), 3); // Closing to try and make the entire thing solid
	
	/// Find the largest contour and get it's details
	vector<vector<Point>> contours; vector<Vec4i> hierarchy; // Variable setup
	findContours(workingImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE); // Find contours
	int largest = getMaxAreaContourId(contours);  // Get index of the largest contour

	/// Square up the working image
	RotatedRect outline = minAreaRect(contours[largest]); // Get the rotated rectangle
	Mat M = getRotationMatrix2D(Point(workingImage.cols/2, workingImage.rows/2), outline.angle, 1);
	warpAffine(workingImage, workingImage, M, workingImage.size(), INTER_LINEAR); // Square up the working image

	/// Get the size and coordinates of the squared up QR code
	findContours(workingImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE); // Find contours
	largest = getMaxAreaContourId(contours);  // Get index of the largest contour

	/// Check which corner is filled in to make the code upright
		// Note: The corners are ~12% of the width, so I used 10% mean value
	Rect qr = boundingRect(contours[largest]);
	int boxSize = round(qr.width * 0.1);
	Rect topLeft(qr.x, qr.y, boxSize, boxSize);
	Rect topRight(qr.x + qr.width - boxSize, qr.y, boxSize, boxSize);
	Rect botLeft(qr.x, qr.y + qr.height - boxSize, boxSize, boxSize);
	Rect botRight(qr.x + qr.width - boxSize, qr.y + qr.height - boxSize, boxSize, boxSize);
	int means[] = { mean(workingImage(topRight))[0], mean(workingImage(topLeft))[0],  // Get the mean value of each element
		mean(workingImage(botLeft))[0], mean(workingImage(botRight))[0] };

	/// Convert the filled corner into a rotation matrix to get it to the top right corner
	int rotationAngle = distance(means, max_element(means, means + sizeof(means)/sizeof(int))) * 90;
	M = getRotationMatrix2D(Point(inImage.cols / 2, inImage.rows / 2), outline.angle - rotationAngle, 1); // Add in original rotation as well
	warpAffine(inImage, inImage, M, inImage.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 255, 255));

	return inImage;
}



String read_qr(Mat inImage) {
/* Read QR
- Takes in an image containing an oriented QR code

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