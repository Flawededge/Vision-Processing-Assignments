// https://github.com/Flawededge/Vision-Processing-Assignments

#include <stdio.h>
#include <chrono>
#include <ctime>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "dirent.h" // Header for reading directories

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

vector<float> EllipticFourierDescriptors(vector<Point>& contour);
int getMaxAreaContourId(vector <vector<cv::Point>> contours);

int main(int argc, char** argv)
{
	/// Read the directory
	DIR* dir;
	struct dirent* ent;
	if ((dir = opendir("C:/hands")) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) {
			// end -> d_name is the current file name
			char *current = ent->d_name;
			//cout << current << "\t-> ";

		/// Read the image and check if it actually read
			Mat image = imread("C:/hands/" + String(current), IMREAD_GRAYSCALE); // Read the image
			if (image.empty()) { // Check if it loaded something
				//cout << "Image empty" << endl;
				continue;
			}

		/// Scale the image to a set height to keep the display looking good
			double scale = 500.0 / double(image.size[0]); 
			resize(image, image, Size(), scale, scale);
			//cout << "Scaled: " << scale << "\t-> ";

			imshow("Working Image", image);

		/// Get the contour of the image and display it
			inRange(image, 10, 255, image); // Threshold image to convert to binary
			imshow("Threshold", image);
			
			// Extract the contours
			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;
			findContours(image, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

			// Get the largest contour and display it on another image
			Mat contourImage(image.size(), CV_8UC3); // Create a blank image for the contours to be drawn on
			contourImage = Scalar(0, 0, 0);
			int id = getMaxAreaContourId(contours);
			drawContours(contourImage, contours, id, Scalar(0, 255, 0), 1);
			imshow("Contours", contourImage);
			
		/// Calculate and show the descriptor
			vector<float> CE;
			CE = EllipticFourierDescriptors(contours[id]); // Get the descriptor

			// Print out the descriptors
			cout << current[0] << ','; // Ge the number this image represents
			for (int i = 0; i < CE.size(); i++) { // Print out each part of the descriptor
				cout << CE[i] << ','; 
			}
			cout << endl;
			waitKey(10);
		}
	}
	else {
		/* could not open directory */
		perror("");
		return EXIT_FAILURE;
	}
	
}

vector<float> EllipticFourierDescriptors(vector<Point> &contour) {
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