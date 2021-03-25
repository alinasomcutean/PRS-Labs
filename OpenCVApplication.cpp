// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <iostream>
#include <fstream>
#include <math.h> 
#include<array> // for array, at() 
#include<tuple> // for get()
#include <list> 
#include <random>
#include <iterator> 
#include <queue>
#include "math.h"

using namespace std;

const int MAXT = 50;

struct peak {
	int theta, ro, hval;
	bool operator < (const peak& o) const {
		return hval > o.hval;
	}
};

struct knn {
	float distance;
	int cls;
	bool operator < (const knn& o) const {
		return distance < o.distance;
	}
};

struct weakLearner {
	int feature_i;
	int threshold;
	int class_label;
	float error;
	int classify(Mat X) {
		if (X.at<float>(feature_i) < threshold) return class_label;
		else return -class_label;
	}
};

struct classifier {
	int T;
	std::vector<float> alphas[MAXT];
	std::vector<weakLearner> hs[MAXT];
	int classify(Mat X) {
		return 0;
	}
};

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = MAX_PATH-val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}

void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

//Lab 1 - Least Mean Squares
std::vector<Point_<float>> readData() {
	FILE *file = fopen("points0.txt", "r");
	int m;
	float x, y;
	fscanf(file, "%d", &m);

	//Create an array with all the points
	std::vector<Point_<float>> points;
	for (int i = 0; i < m; i++) {
		fscanf(file, "%f%f", &x, &y);
		Point_<float> p;
		p.x = x;
		p.y = y;
		points.push_back(p);
	}
	fclose(file);
	return points;
}

Mat_<uchar> drawPoints(std::vector<Point_<float>> points) {
	Mat_<uchar> img(500, 500, CV_8UC3);

	//Make the background white
	for (int i = 0; i < 500; i++) {
		for (int j = 0; j < 500; j++) {
			img(i, j) = (255, 255, 255);
		}
	}

	//Draw a small circle for every point
	for (Point_<float> p : points) {
		circle(img, p, 2.0f, (0, 0, 0));
	}

	imshow("Image", img);
	waitKey();
	return img;
}

float* computeTeta(Mat_<uchar> img, std::vector<Point_<float>> points) {
	float teta0 = 0, teta1 = 0;
	float sumY = 0, sumX = 0, sumSquareX = 0, sumXY = 0;
	int n = points.size();
	
	//Compute the sums for teta0 and teta1
	for (Point_<float> p : points) {
		sumXY += p.x * p.y;
		sumX += p.x;
		sumY += p.y;
		sumSquareX += p.x * p.x;
	}

	//Compute teta0 and teta1
	teta1 = (n * sumXY - sumX * sumY) / (n * sumSquareX - sumX * sumX);
	teta0 = (sumY - teta1 * sumX) / n;

	float teta[2];
	teta[0] = teta0;
	teta[1] = teta1;

	printf("Teta 0: %f \nTeta 1: %f", teta[0], teta[1]);
	int a;
	cin >> a;

	return teta;
}

//Lab 2 - RANSAC, fitting a line to a set of points
std::vector<Point_<int>> constructInputPointSet(Mat_<uchar> src) {
	int height = src.rows;
	int width = src.cols;

	//Create a vector to store all the points from the image
	std::vector<Point_<int>> points;

	//Traverse the image to find all the points
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			//If a black point is found
			if (src(i, j) == 0) {
				Point_<int> p;
				p.x = j;
				p.y = i;
				//Put it in the created vector
				points.push_back(p);
			}
		}
	}

	return points;
}

void ransacMethod() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		//Read an image
		Mat_<uchar> src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		
		//Construct the point set from the input image
		std::vector<Point_<int>> points = constructInputPointSet(src);

		float p = 0.99f, q = 0.3f;
		int t = 10, s = 2;
		int pointsNo = points.size();

		//Calculate the parameters N (max number of iteration) and T
		float N = 0.0f, T = 0.0f;
		N = (log(1 - p)) / log(1 - pow(q, s));
		T = q * pointsNo;
		printf("PointsNo = %d, N = %f, T = %f\n", pointsNo, N, T);

		//Apply Ransac method
		int iteration = 0;
		int a = 0, b = 0, c = 0, inliers = 0;
		Point_<int> finalP1, finalP2;

		while (iteration <= N || inliers <= T) {
			//a: choose 2 points randomnly;
			int p1Position = rand() % pointsNo;
			int p2Position = rand() % pointsNo;

			Point_<int> p1 = points.at(p1Position);
			Point_<int> p2 = points.at(p2Position);

			//b: determine the line equation passing through the selected points
			int currentA, currentB, currentC;
			currentA = p1.y - p2.y;
			currentB = p2.x - p1.x;
			currentC = p1.x * p2.y - p2.x * p1.y;

			int currentInliers = 0;

			for (int i = 0; i < pointsNo; i++) {
				Point_<int> currentPoint = points.at(i);

				//c: find the distances of each point to the line
				float distance = abs(currentA * currentPoint.x + currentB * currentPoint.y + currentC) / sqrt(currentA * currentA + currentB * currentB);
				
				//d: count the number of inliers
				if (distance < t) {
					currentInliers++;
				}
			}

			//e: find the highest number of inliers so far
			//save the parameters a,b,c if the current line has the highest no of inliers
			if (currentInliers > inliers) {
				inliers = currentInliers;
				a = currentA;
				b = currentB;
				c = currentC;
				finalP1 = p1;
				finalP2 = p2;
			}

			//line(src, p1, p2, Scalar(0, 0, 255));
			iteration++;
		}

		printf("a = %d, b = %d, c = %d, inliers = %d", a, b, c, inliers);
		line(src, finalP1, finalP2, Scalar(0, 0, 255));
		imshow("Image", src);
		//imshow("dest", dst);
		waitKey(0);
	}
}

//Lab 3 - Hough Transform for line detection
Mat_<uchar> cannyAlgorithm(Mat src) {
	int height = src.rows;
	int width = src.cols;

	Mat dst(height, width, CV_8UC1);
	dst.setTo(0);

	//Blur the image with a filter of kernel size 3
	blur(src, dst, Size(3, 3));
	Canny(dst, dst, 0, 0, 3);

	imshow("Edge", dst);

	return dst;
}

void computeHoughAccumulator() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst(height, width, CV_8UC3);

		//Compute the edge detection
		Mat_<uchar> edge_image = cannyAlgorithm(src);

		//Compute the image diagonal
		int roMax = sqrt(height * height + width * width);

		//Define the accumulator
		Mat Hough(roMax + 1, 360, CV_32SC1);

		//Initialize the accumulator
		Hough.setTo(0);

		//Iterate on every pixel in the image
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				dst.at<Vec3b>(i, j)[0] = src.at<uchar>(i, j);
				dst.at<Vec3b>(i, j)[1] = src.at<uchar>(i, j);
				dst.at<Vec3b>(i, j)[2] = src.at<uchar>(i, j);
				//Check if the pixel is on the edge (pixel is white)
				if (edge_image(i,j) == 255) {
					for (int k = 0; k < 360; k++) {
						//Compute theta
						float theta = k * CV_PI / 180;
						//Compute ro
						int ro = j * cos(theta) + i * sin(theta);
						//Check if the computed ro is in the right interval
						if (ro >= 0 && ro <= roMax) {
							//If yes, increment the cell in Hough accumulator
							Hough.at<int>(ro, k)++;
						}
					}
				}
			}
		}

		//Find the max value from the accumulator
		int maxHough = 0;
		for (int i = 0; i < roMax; i++) {
			for (int j = 0; j < 360; j++) {
				int currentValue = Hough.at<int>(i, j);
				if (currentValue > maxHough) {
					maxHough = currentValue;
				}
			}
		}

		//Display the Hough accumulator
		Mat houghImg;
		Hough.convertTo(houghImg, CV_8UC1, 255.f/ maxHough);
		imshow("Hough accumulator", houghImg);

		//Find the local max (peaks) in the accumulator
		std::vector<peak> localMax;

		for (int i = 1; i < roMax; i++) {
			for (int j = 1; j < 359; j++) {
				//Check if the the element is local max
				int currentValue = Hough.at<int>(i, j);
				if (currentValue > Hough.at<int>(i - 1, j - 1) && currentValue > Hough.at<int>(i - 1, j) && currentValue > Hough.at<int>(i - 1, j + 1) && 
					currentValue > Hough.at<int>(i, j - 1) && currentValue > Hough.at<int>(i, j + 1) && 
					currentValue > Hough.at<int>(i + 1, j - 1) && currentValue > Hough.at<int>(i + 1, j) && currentValue > Hough.at<int>(i + 1, j + 1)) {
					//Check if the element is greater then a threshold = 20
					if (currentValue > 20) {
						peak p;
						p.hval = currentValue;
						p.ro = i;
						p.theta = j;
						localMax.push_back(p);
					}
				}
			}
		}

		//Sort the local max values kept
		std::sort(localMax.begin(), localMax.end());

		printf("Local max size: %d\n", localMax.size());

		int x1 = 0, x2 = width - 1;
		int y1, y2;
		for (int i = 0; i < 10; i++) {
			peak line1 = localMax.at(i);
			y1 = line1.ro / sin(line1.theta * CV_PI / 180);
			y2 = (line1.ro - x2 * cos(line1.theta * CV_PI / 180)) / sin(line1.theta * CV_PI / 180);

			Point p1, p2;
			p1.x = x1, p1.y = y1;
			p2.x = x2, p2.y = y2;

			line(dst, p1, p2, Scalar(0, 204, 0));
		}
		imshow("Line", src);
		imshow("dst", dst);
		waitKey(0);
	}
}

//Lab 4 - Distance Transform (DT) and Pattern Matching using DT
Mat distanceTransformAlg() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		//Select a mask and split it into 2 parts
		int di[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
		int dj[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };
		int weight[8] = { 3, 2, 3, 2, 2, 3, 2, 3 };
		
		//Scan the DT with submask1
		for (int i = 1; i < height; i++) {
			for (int j = 1; j < width - 1; j++) {
				uchar minim = dst.at<uchar>(i,j);
				for (int k = 0; k < 4; k++) {
					int pixel = dst.at<uchar>(i + di[k], j + dj[k]) + weight[k];
					if (pixel < minim) {
						minim = pixel;
					}
				}
				dst.at<uchar>(i, j) = minim;
			}
		}

		//Scan the DT with submask2
		for (int i = height - 2; i >= 0; i--) {
			for (int j = width - 2; j > 0; j--) {
				uchar minim = dst.at<uchar>(i, j);
				for (int k = 4; k < 8; k++) {
					int pixel = dst.at<uchar>(i + di[k], j + dj[k]) + weight[k];
					if (pixel < minim) {
						minim = pixel;
					}
				}
				dst.at<uchar>(i, j) = minim;
			}
		}

		imshow("Source img", src);
		//imshow("Dest img", dst);
		//waitKey(0);

		return dst;
	}
}

float computeMatchingScore(Mat src, Mat img) {
	int height = src.rows;
	int width = src.cols;

	float sum = 0.0f;
	int count = 0;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (img.at<uchar>(i, j) == 0) {
				count++;
				sum += src.at<uchar>(i, j);
			}
		}
	}

	return sum / count;
}

void compareMatchingScores() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		//Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat templateObj = imread(fname, IMREAD_GRAYSCALE);
		Mat templateImg = distanceTransformAlg();

		printf("Matching score: %f\n", computeMatchingScore(templateImg, templateObj));

		imshow("Image dst", templateImg);
		imshow("Object", templateObj);
		waitKey(0);
	}
}

//Lab 5 - Statistical Data Analysis
Mat computeIntesityMatrix() {
	char folder[256] = "Images/prs_res_Statistics";
	char fname[256];
	Mat intensity(400, 361, CV_8UC1);

	for (int i = 1; i <= 400; i++) {
		sprintf(fname, "%s/face%05d.bmp", folder, i);
		Mat img = imread(fname, 0);
		int height = img.rows;
		int width = img.cols;
		int n = 0;

		for (int h = 0; h < height; h++) {
			for (int w = 0; w < width; w++) {
				intensity.at<uchar>(i - 1, n) = img.at<uchar>(h, w);
				n++;
			}
		}
	}

	return intensity;
}

std::vector<float> computeMeanValue() {
	std::vector<float> meanValues;
	Mat intensity = computeIntesityMatrix();
	FILE *file;
	file = fopen("Files/MeanValues.csv", "w");

	if (file == NULL) {
		printf("Error opening the file\n");
		exit(1);
	}

	for (int j = 0; j < 361; j++) {
		int sum = 0;
		for (int i = 0; i < 400; i++) {
			sum += intensity.at<uchar>(i, j);
		}
		float meanValue = sum / 400.0f;
		meanValues.push_back(meanValue);
		fprintf(file, "%f, ", meanValue);
	}

	fclose(file);
	return meanValues;
}

std::vector<float> computeStandardDeviation() {
	std::vector<float> standardDeviations;
	Mat intensity = computeIntesityMatrix();
	std::vector<float> meanValues = computeMeanValue();

	for (int j = 0; j < 361; j++) {
		float sum = 0;
		for (int i = 0; i < 400; i++) {
			sum += pow((intensity.at<uchar>(i,j) - meanValues.at(j)), 2);
		}
		standardDeviations.push_back(sqrt(sum / 400));
	}
	return standardDeviations;
}

Mat computeCovarianceMatrix() {
	Mat covariance(361, 361, CV_32FC1);
	Mat intensity = computeIntesityMatrix();
	vector<float> meanValues = computeMeanValue();
	vector<float> standardDeviations = computeStandardDeviation();

	for (int i = 0; i < 361; i++) {
		for (int j = 0; j < 361; j++) {
			if (i == j) {
				covariance.at<float>(i, j) = standardDeviations.at(i) * standardDeviations.at(i);
			}
			else {
				float sum = 0.0f;
				for (int k = 0; k < 400; k++) {
					sum += (intensity.at<uchar>(k, i) - meanValues.at(i)) * (intensity.at<uchar>(k, j) - meanValues.at(j));
				}
				covariance.at<float>(i, j) = sum / 400.0f;
			}
		}
	}

	return covariance;
}

std::vector<float> computeCorrelationCoeff() {
	std::vector<float> correlationCoeff;
	Mat covariance = computeCovarianceMatrix();
	std::vector<float> standardDeviation = computeStandardDeviation();

	FILE *file;
	file = fopen("Files/CorrelationCoeff.csv", "w");

	if (file == NULL) {
		printf("Error opening the file\n");
		exit(1);
	}

	for (int i = 0; i < 361; i++) {
		for (int j = 0; j < 361; j++) {
			float value = covariance.at<float>(i, j) / (standardDeviation.at(i) * standardDeviation.at(j));
			correlationCoeff.push_back(value);
			fprintf(file, "%f, ", value);
		}
		fprintf(file, "\n");
	}

	fclose(file);
	return correlationCoeff;
}

void correlationChart(){
	Mat chart(256, 256, CV_8UC1);
	Mat intensity = computeIntesityMatrix();
	chart.setTo(255);

	int i1, i2, j1, j2;
	printf("Insert 2 features pairs between 0 and 360\n");
	printf("i1 = ");
	cin >> i1;
	printf("i2 = ");
	cin >> i2;
	printf("j1 = ");
	cin >> j1;
	printf("j2 = ");
	cin >> j2;

	//transform to a single value using the row-major order
	int index1 = i1 * 19 + j1;
	int index2 = i2 * 19 + j2;

	for (int k = 0; k < 400; k++) {
		chart.at<uchar>(intensity.at<uchar>(k, index2), intensity.at<uchar>(k, index1)) = 0;
	}

	imshow("chart", chart);
	waitKey(0);
}

void plotDensityFunction() {
	std::vector<float> meanValues = computeMeanValue();
	std::vector<float> standardDeviation = computeStandardDeviation();

	Mat img(256, 256, CV_8UC1);
	img.setTo(255);

	int feature;
	printf("Insert the feature number: ");
	cin >> feature;

	for (int x = 0; x < 256; x++) {
		float numarator = pow(x - meanValues.at(feature), 2);
		float numitor = 2 * pow(standardDeviation.at(feature), 2);
		float value = (1 / (sqrt(2 * PI) * standardDeviation.at(feature))) * exp((-numarator) / numitor);
		int fx = value * 10000;
		img.at<uchar>(x, fx) = 0;
	}

	imshow("img", img);
	waitKey(0);
}

//Lab 7 - Principal Component Analysis
std::vector<double> meanValuesPca(Mat X) {
	std::vector<double> meanVector;

	//Travers the matrix in columns and compute the average for each column
	for (int j = 0; j < X.cols; j++) {
		double sum = 0.0;
		for (int i = 0; i < X.rows; i++) {
			sum += X.at<double>(i, j);
		}
		meanVector.push_back(sum / X.rows);
	}

	return meanVector;
}

void pca() {
	FILE *file = fopen("pca2d.txt", "r");
	int n, d;

	//STEP 1:
	//Read the number of points and the dimensionality of the data
	fscanf(file, "%d%d", &n, &d);
	Mat X(n, d, CV_64FC1);
	
	//Read the input data and initialize the matrix X
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < d; j++) {
			fscanf(file, "%lf", &X.at<double>(i,j));
		}
	}

	//STEP 2:
	//Compute the mean values for the input data
	std::vector<double> meanVector = meanValuesPca(X);
	
	//Travers the input data
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < d; j++) {
			//For each element substract the corresponding mean value
			X.at<double>(i, j) -= meanVector[j];
		}
	}

	//STEP 3:
	//Compute the covariance matrix
	Mat C(n, d, CV_64FC1);
	C = (X.t() * X) / (n - 1);

	//STEP 4:
	//Perform the eigenvalue decompositions
	Mat Lambda, Q;
	eigen(C, Lambda, Q);
	Q = Q.t();

	//STEP 5:
	//Print the eigenvalues
	printf("Eigenvalues: ");
	for (int i = 0; i < Lambda.rows; i++) {
		for (int j = 0; j < Lambda.cols; j++) {
			printf("%.2lf ", Lambda.at<double>(i, j));
		}
	}
	printf("\n");

	//STEP 6:
	//Compute the PCA coefficients
	Mat Xcoeff = X * Q;

	//STEP 6.1:
	//Compute PCA approximations of order k using the first k columns from Q (k=1)
	int k = 1;
	Rect submat(0, 0, k, Q.rows);
	Mat Qk = Q(submat);
	Mat Xk = X * Qk * Qk.t();

	//STEP 7:
	double absoluteErr = 0;
	for (int j = 0; j < d; j++) {
		for (int i = 0; i < n; i++) {
			absoluteErr = absoluteErr + abs(X.at<double>(i, j) - Xk.at<double>(i, j));
		}
	}
	printf("Absolut error is %.2lf", absoluteErr / (d * n));

	//STEP 8:
	//Display an image with the first 2 columns from the coefficient matrix
	//Initialize the final image
	Mat img(300, 300, CV_8UC1);
	img.setTo(255);

	//Find the min for both columns
	double min0 = Xcoeff.at<double>(0, 0);
	double min1 = Xcoeff.at<double>(0, 0);

	for (int i = 1; i < n; i++) {
		if (Xcoeff.at<double>(i, 0) < min0) {
			min0 = Xcoeff.at<double>(i, 0);
		}
		if (Xcoeff.at<double>(i, 1) < min1) {
			min1 = Xcoeff.at<double>(i, 1);
		}
	}

	//Draw the final image
	for (int i = 0; i < n; i++) {
		img.at<uchar>(Xcoeff.at<double>(i, 0) - min0, Xcoeff.at<double>(i, 1) - min1) = 0;
	}

	imshow("Image", img);
	waitKey(0);
}

//Lab 8 - K-means clustering
float computeDistanceBetweenPoints(Point p1, Point p2) {
	int a = p1.x - p2.x;
	int b = p1.y - p2.y;
	int value = a * a + b * b;
	return sqrt(value);
}

std::vector<Point_<int>> initializeInputSetPoint(Mat src, int height, int width) {
	std::vector<Point_<int>> xPoints;
	//Compute the input set for the algorithm
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			//If the pixel is black, add it in the vector of points
			if (src.at<uchar>(i, j) == 0) {
				Point p;
				p.x = i;
				p.y = j;
				xPoints.push_back(p);
			}
		}
	}

	return xPoints;
}

Mat assignPoints(Mat dst, std::vector<Point_<int>> points, std::vector<Vec3b> colors, std::vector<Point_<int>> centers, int k) {
	//Step 2: assignment
	//For each point in the input set
	for (int i = 0; i < points.size(); i++) {
		Point point = points.at(i);
		//Find the closest cluster center and its color
		float minDistance = computeDistanceBetweenPoints(point, centers.at(0));
		Vec3b color = colors.at(0);
		for (int j = 1; j < k; j++) {
			float currentDistance = computeDistanceBetweenPoints(point, centers.at(j));
			if (currentDistance < minDistance) {
				minDistance = currentDistance;
				color = colors.at(j);
			}
		}
		//Put the results in the final image
		dst.at<Vec3b>(point.x, point.y) = color;
	}

	return dst;
}

Point computeNewCenter(std::vector<Vec3b> colors, int colorNo, Mat dst) {
	int sumX = 0, sumY = 0, pointNo = 0;
	Vec3b currentColor = colors.at(colorNo);

	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {
			//Compute the sum for each coordination of the points
			if (dst.at<Vec3b>(i, j) == currentColor) {
				sumX += i;
				sumY += j;
				pointNo++;
			}
		}
	}

	Point_<int> p;
	p.x = sumX / pointNo;
	p.y = sumY / pointNo;

	return p;
}

Mat drawVoronoiImage(std::vector<Vec3b> colors, std::vector<Point_<int>> centers, int k, Mat voronoi) {
	//For each point in the image
	for (int i = 0; i < voronoi.rows; i++) {
		for (int j = 0; j < voronoi.cols; j++) {
			Point point;
			point.x = i;
			point.y = j;
			//Compute the disance to each cluster's center
			float minDistance = computeDistanceBetweenPoints(point, centers.at(0));
			Vec3b color = colors.at(0);
			for (int j = 1; j < k; j++) {
				float currentDistance = computeDistanceBetweenPoints(point, centers.at(j));
				//Keep the minimum distance and its color
				if (currentDistance < minDistance) {
					minDistance = currentDistance;
					color = colors.at(j);
				}
			}
			//Put the color of the minimum distance in the image
			voronoi.at<Vec3b>(i, j) = color;
		}
	}

	return voronoi;
}

void kMeansForPoints() {
	char fname[MAX_PATH];

	//Read the number of clusters, k
	int k;
	printf("K = ");
	scanf("%d", &k);

	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		Mat dst(height, width, CV_8UC3);
		dst.setTo(255);
		std::vector<Point_<int>> xPoints;

		//Step 1: initialization
		//Compute the input set for the algorithm
		xPoints = initializeInputSetPoint(src, height, width);

		int xPointsSize = xPoints.size();
		default_random_engine gen;
		uniform_int_distribution<int> c(50, 180);
		uniform_int_distribution<int> distribution(0, xPointsSize - 1);

		//Assign random colors to clusters
		std::vector<Vec3b> colors;
		//Get the randomnly selected K centers
		std::vector<Point_<int>> centers;

		for (int i = 0; i < k; i++) {
			colors.push_back({ (uchar)c(gen), (uchar)c(gen), (uchar)c(gen) });
			Point_<int> p = xPoints.at(distribution(gen));
			centers.push_back(p);
			dst.at<Vec3b>(p.x, p.y) = colors.at(i);
		}

		bool change = true;

		//While there is a change in finding the center of the clusters
		while (change) {
			int countChanges = 0;
			//Step 2: assignment
			//Assign a cluster for each point in the input set
			dst = assignPoints(dst, xPoints, colors, centers, k);

			//Step 3: update
			//Recalculate the cluster centers for each one
			for (int colorNo = 0; colorNo < k; colorNo++) {
				Point_<int> p = computeNewCenter(colors, colorNo, dst);

				//Find if there is a change from the previous step
				if (centers.at(colorNo).x != p.x || centers.at(colorNo).y != p.y) {
					centers.at(colorNo) = p;
					countChanges++;
				} 
			}

			if (countChanges == 0) {
				change = false;
			}
		}

		//Draw the Voroni tessellation
		Mat voronoi(height, width, CV_8UC3);
		voronoi.setTo(255);
		voronoi = drawVoronoiImage(colors, centers, k, voronoi);

		Mat cent(height, width, CV_8UC3);
		cent.setTo(255);
		int di[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
		int dj[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };

		for (int i = 0; i < k-1; i++) {
			Point currentCenter = centers.at(i);
			Vec3b currentColor = colors.at(i);
			for (int j = 0; j < 8; j++) {
				cent.at<Vec3b>(currentCenter.x + di[j], currentCenter.y + dj[j]) = currentColor;
			}
		}

		for (int j = 0; j < 8; j++) {
			cent.at<Vec3b>(centers.at(k-1).x + di[j], centers.at(k - 1).y + dj[j]) = (255,0,0);
		}

		imshow("Initial image", src);
		imshow("K-means image", dst);
		imshow("Voroni image", voronoi);
		imshow("Centers image", cent);
		waitKey(0);
	}
}

std::vector<int> initializeGrayInputSet(Mat img) {
	std::vector<int> inputSet;
	int heigh = img.rows;
	int width = img.cols;

	for (int i = 0; i < heigh; i++) {
		for (int j = 0; j < width; j++) {
			if (std::find(inputSet.begin(), inputSet.end(), (int) img.at<uchar>(i, j)) == inputSet.end()) {
				inputSet.push_back((int) img.at<uchar>(i,j));
			}
		}
	}

	return inputSet;	
}

void kMeansGrayscale() {
	char fname[MAX_PATH];
	int k;
	printf("K = ");
	scanf("%d", &k);

	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst(height, width, CV_8UC1);
		src.copyTo(dst);

		//Step 1: initialization
		//Create input set
		std::vector<int> inputSet = initializeGrayInputSet(src);

		int inputSetSize = inputSet.size();
		std::random_device rd; 
		default_random_engine gen(rd());
		uniform_int_distribution<int> c(50, 180);
		uniform_int_distribution<int> distribution(0, inputSetSize - 1);

		//Get the randomnly selected K centers
		std::vector<int> centers;

		//select k random intensities from the input set
		for (int i = 0; i < k; i++) {
			int intensity = inputSet.at(distribution(gen));
			centers.push_back(intensity);
		}

		//go through the image
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				//select the first intensity as the minim
				int minIntensity = centers.at(0);

				//go through the vector of the intensities
				for (int value = 1; value < k; value++) {
					int diff = abs(src.at<uchar>(i, j) - centers.at(value));
					if (diff < minIntensity) {
						minIntensity = centers.at(value);
					}
				}

				//update the intensity value in the destination image
				dst.at<uchar>(i, j) = minIntensity;
			}
		}

		imshow("Initial image", src);
		imshow("K-means image grayscale", dst);
		waitKey(0);
	}
}

std::vector<Vec3b> initializeColorInputSet(Mat img) {
	std::vector<Vec3b> inputSet;
	int heigh = img.rows;
	int width = img.cols;

	for (int i = 0; i < heigh; i++) {
		for (int j = 0; j < width; j++) {
			if (std::find(inputSet.begin(), inputSet.end(), img.at<Vec3b>(i, j)) == inputSet.end()) {
				inputSet.push_back(img.at<Vec3b>(i, j));
			}
		}
	}

	return inputSet;
}

void kMeansColor() {
	char fname[MAX_PATH];
	int k;
	printf("K = ");
	scanf("%d", &k);

	while (openFileDlg(fname)) {
		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat dst(height, width, CV_8UC3);
		src.copyTo(dst);

		//Step 1: initialization
		//Create input set
		std::vector<Vec3b> inputSet = initializeColorInputSet(src);

		int inputSetSize = inputSet.size();
		std::random_device rd;
		default_random_engine gen(rd());
		uniform_int_distribution<int> c(50, 180);
		uniform_int_distribution<int> distribution(0, inputSetSize - 1);

		//Get the randomnly selected K centers
		std::vector<Vec3b> centers;

		//select k random colors from the input set
		for (int i = 0; i < k; i++) {
			Vec3b color = inputSet.at(distribution(gen));
			centers.push_back(color);
		}

		for (int i = 0; i < centers.size(); i++) {
			printf("%d %d %d\n", centers.at(i)[0], centers.at(i)[1], centers.at(i)[2]);
		}

		//go through the image
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				//select the first intensity as the minim
				int b = centers.at(0)[0] - src.at<Vec3b>(i, j)[0];
				int g = centers.at(0)[1] - src.at<Vec3b>(i, j)[1];
				int r = centers.at(0)[2] - src.at<Vec3b>(i, j)[2];
				int minIntensity = b * b + g * g + r * r;
				Vec3b minValue = centers.at(0);

				//go through the vector of the intensities
				for (int value = 1; value < k; value++) {
					int currB = centers.at(value)[0] - src.at<Vec3b>(i, j)[0];
					int currG = centers.at(value)[1] - src.at<Vec3b>(i, j)[1];
					int currR = centers.at(value)[2] - src.at<Vec3b>(i, j)[2];
					int diff = currB * currB + currG * currG + currR * currR;
					if (diff < minIntensity) {
						minValue = centers.at(value);
					}
				}

				//update the intensity value in the destination image
				dst.at<Vec3b>(i, j) = minValue;
			}
		}

		imshow("Initial image", src);
		imshow("K-means image color", dst);
		waitKey(0);
	}
}

//Lab 9 - Naive Bayesian Classifier
Mat computeLikelihood(int c, int d, Mat trainingMatrix, Mat y) {
	Mat likelihood(c, d, CV_64FC1);
	int height = trainingMatrix.rows;
	int width = trainingMatrix.cols;
	
	std::vector<int> count(c, 0);

	for (int w = 0; w < width; w++) {
		std::vector<int> count(c, 0);
		//Count hte number of pixels from each class equal with 255
		for (int h = 0; h < height; h++) {
			if (trainingMatrix.at<uchar>(h, w) == 255) {
				count[y.at<uchar>(h, 0)]++;
			}
		}

		//Compute for each class, for each feature likelihood for 255
		for (int i = 0; i < c; i++) {
			likelihood.at<double>(i, w) = (count[i] + 1) / (double) (y.rows / c);
		}
	}

	return likelihood;
}

Mat computeFeatureMatrix(Mat img) {
	int height = img.rows;
	int width = img.cols;

	//Declare feature matrix, which has only 1 row (1 image)
	Mat features(1, height * width, CV_8UC1);
	
	//Compute the feature matrix
	for (int h = 0; h < height; h++) {
		for (int w = 0; w < width; w++) {
			features.at<uchar>(0, h * 28 + w) = (int) img.at<uchar>(h, w);
		}
	}

	return features;
}

int classifyBayes(Mat img, Mat priors, Mat likelihood, int c, int d) {
	int height = img.rows;
	int width = img.cols;
	Mat features(1, d, CV_8UC1);

	//Compute the feature matrix for the new image
	features = computeFeatureMatrix(img);

	//Compute the log posterior of each class
	Mat prob(c, 1, CV_64FC1, 0.0f);

	//For each class
	for (int i = 0; i < c; i++) {
		//Get the initial value of the posterior;
		prob.at<double>(i, 0) = log(priors.at<double>(i, 0));
		for (int j = 0; j < d; j++) {
			//Update the posterior value accordingly
			if (features.at<uchar>(0, j) == 255) {
				prob.at<double>(i, 0) += log(likelihood.at<double>(i, j));
			}
			else {
				prob.at<double>(i, 0) += log(1 - likelihood.at<double>(i, j));
			}
		}
	}

	//Get the initial maxim and its class
	double maxim = prob.at<double>(0, 0);
	int cls = 0;

	//Check if another class has a higher maxim
	for (int i = 1; i < c; i++) {
		double current = prob.at<double>(i, 0);
		if (current > maxim) {
			maxim = current;
			cls = i;
		}
	}

	return cls;
}

void computeTrainingMatrix() {
	char folder[256] = "Images/prs_res_Bayes";
	char fname[256];

	int num_samples = 200;
	int d = 784;
	//2 classes (0 and 1)
	int c = 2;

	//feature matrix
	Mat trainingMatrix(num_samples, d, CV_8UC1);

	//class of each feature
	Mat y(num_samples, 1, CV_8UC1);

	//samples that belongs to each class
	Mat priors(c, 1, CV_64FC1);
	Mat likelihood(c, d, CV_64FC1);

	//Read images for the first 2 classes
	for (int k = 0; k < c; k++) {
		//For each class read the first 100 images
		int index = 0;

		while (index < 100) {
			//Put the class of each sample
			y.at<uchar>(100 * k + index, 0) = k;

			//Read the image
			sprintf(fname, "%s/train/%d/%06d.png", folder, k, index);
			Mat img = imread(fname, 0);
			if (img.cols == 0) {
				break;
			}

			threshold(img, img, 128, 255, CV_THRESH_BINARY);
			
			//Travers the image in order to save it in the training matrix
			for (int row = 0; row < 28; row++) {
				for (int col = 0; col < 28; col++) {
					trainingMatrix.at<uchar>(100 * k + index, row * 28 + col) = img.at<uchar>(row, col);
				}
			}

			index++;
		}

		//Compute the prior for each class
		priors.at<double>(k, 0) = index / (float)num_samples;
	}

	likelihood = computeLikelihood(c, d, trainingMatrix, y);

	for (int i = 0; i < c; i++) {
		for (int j = 0; j < d; j++) {
			printf("%.2f ", likelihood.at<double>(i, j));
		}
		printf("\n");
	}
;
	while (openFileDlg(fname)) {
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		threshold(img, img, 128, 255, CV_THRESH_BINARY);

		int cls = classifyBayes(img, priors, likelihood, c, d);

		imshow("Image to classify", img);
		printf("Class: %d\n", cls);
		waitKey(0);
	}
}

//Lab 10 - k-nearest neighbors classifier
void showHistogram(const std::string& name, std::vector<int> hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
	waitKey(0);
}

std::vector<int> colorHistogram(Mat img, int m) {
	int height = img.rows;
	int width = img.cols;

	//Create a vector for the histogram of a color image of size 3*256 because there are 3 channels: B, G, R
	std::vector<int> hist(3 * m, 0);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			Vec3b pixel = img.at<Vec3b>(i, j);
			hist[pixel[0]]++;
			hist[m + pixel[1]]++;
			hist[2 * m + pixel[2]]++;
		}
	}

	//showHistogram("hist", hist, 3*m, 500);
	return hist;
}

void createMatrixX(Mat& x, int m, int rowX, Mat img) {
	//Calculate the color histogram
	std::vector<int> hist = colorHistogram(img, m);
	for (int d = 0; d < hist.size(); d++) {
		x.at<float>(rowX, d) = hist[d];
	}
}

void computeMatrixXandY(Mat& x, Mat& y, int m) {
	char fname[MAX_PATH];
	char folder[256] = "Images/prs_res_KNN";

	//Define the class names
	const int nrClasses = 6;
	char classes[nrClasses][10] = { "beach", "city", "desert", "forest", "landscape", "snow" };
	
	int rowX = 0;
	for (int c = 0; c < nrClasses; c++) {
		//Read all the images from the classes, calculate the histogram and insert values in x
		int fileNr = 0;

		while (1) {
			//Read the image
			sprintf(fname, "%s/train/%s/%06d.jpeg", folder, classes[c], fileNr++);
			Mat img = imread(fname);
			if (img.cols == 0) break;

			createMatrixX(x, m, rowX, img);

			//Save the label for the image
			y.at<uchar>(rowX) = c;

			//Go to the next image
			rowX++;
		}
	}
}

void distanceFunction(Mat testX, Mat x, Mat y, std::vector<knn>& dist) {
	for (int i = 0; i < x.rows; i++) {
		float sum = 0.0f;
		for (int j = 0; j < x.cols; j++) {
			float dif = pow(testX.at<float>(0, j) - x.at<float>(i, j), 2);
			sum += dif;
		}

		knn d;
		d.distance = sqrt(sum);
		d.cls = y.at<uchar>(i);

		dist.push_back(d);
	}
}

int knnClassifier(Mat testImg, int m, Mat x, Mat y) {
	const int classesNo = 6;
	char classes[classesNo][10] = { "beach", "city", "desert", "forest", "landscape", "snow" };

	Mat testX(1, 3 * m, CV_32FC1);
	createMatrixX(testX, m, 0, testImg);

	//Compute distance between feature vector for the test image and the matrix with features of all the images
	std::vector<knn> dist;
	distanceFunction(testX, x, y, dist);

	//Sort the vector in ascending order
	std::sort(dist.begin(), dist.end());

	//Read the value for k
	int k = 20;
	/*printf("%s %d\n%s", "Choose a value smaller than", dist.size(), "k = ");
	scanf("%d", &k);*/

	std::vector<int> c(classesNo, 0);
	
	//For first k distances, count how many times each class appears
	for (int i = 0; i < k; i++) {
		c[dist.at(i).cls]++;
	}

	//Find the class with the most occurences
	int maxim = 0;
	int cls = -1;
	for (int i = 0; i < classesNo; i++) {
		if (c[i] > maxim) {
			maxim = c[i];
			cls = i;
		}
	}
	cout << cls << "\n";
	
	if (cls != -1) {	
		printf("%s\n", classes[cls]);
	}
	
	return cls;
}

void confussionMatrixAndAccuracy() {
	char fname[MAX_PATH];
	char folder[256] = "Images/prs_res_KNN";

	//Read the number of bins
	int m;
	printf("m = ");
	scanf("%d", &m);

	//Define the class names
	const int nrClasses = 6;
	char classes[nrClasses][10] = { "beach", "city", "desert", "forest", "landscape", "snow" };

	//Allocate the feature matrix and the label vector
	int imgNo = 672;
	Mat x(imgNo, 3 * m, CV_32FC1);
	Mat y(imgNo, 1, CV_8UC1);

	//Compute feature matrix x and label matrix y
	computeMatrixXandY(x, y, m);

	Mat conff(nrClasses, nrClasses, CV_8UC1);
	conff.setTo(0);

	//Variables for accuracy
	int totalImgTest = 0;
	int correctPredicted = 0;

	for (int c = 0; c < nrClasses; c++) {
		//Read all the images from the classes, calculate the histogram and insert values in x
		int fileNr = 0;

		while (1) {
			//Read the image
			sprintf(fname, "%s/test/%s/%06d.jpeg", folder, classes[c], fileNr++);
			Mat img = imread(fname);
			if (img.cols == 0) break;

			int resultClass = knnClassifier(img, m, x, y);

			conff.at<uchar>(c, resultClass)++;
			totalImgTest++;
		}
	}

	for (int i = 0; i < nrClasses; i++) {
		correctPredicted += conff.at<uchar>(i, i);
	}

	printf("\nConfusion matrix:\n");

	for (int i = 0; i < nrClasses; i++) {
		for (int j = 0; j < nrClasses; j++) {
			printf("%d ", conff.at<uchar>(i, j));
		}
		printf("\n");
	}

	int accuracy = correctPredicted * 100 / totalImgTest;
	
	printf("\nAccuracy: %d%", accuracy);

	int f;
	cin >> f;
}

//Lab 11 - linear classifiers and the perceptron algorithm
void constructTrainingSet(Mat img, std::vector<std::vector<int>> &X, std::vector<int> &Y) {
	int height = img.rows;
	int width = img.cols;

	for (int h = 0; h < height; h++) {
		for (int w = 0; w < width; w++) {
			Vec3b pixel = img.at<Vec3b>(h, w);
			//check if the pixel is blue
			if (pixel[0] == 255 && pixel[1] == 0 && pixel[2] == 0) {
				std::vector<int> row;
				row.push_back(1);
				row.push_back(w);
				row.push_back(h);
				X.push_back(row);
				Y.push_back(-1);
			}
			if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 255) {
				std::vector<int> row;
				row.push_back(1);
				row.push_back(w);
				row.push_back(h);
				X.push_back(row);
				Y.push_back(1);
			}
		}
	}
}

void drawLine(Mat img, std::vector<float> w) {
	Point p1, p2;

	//line equation: ax + by + c = 0
	//a = w[1], b = w[2], c = w[0]

	if (w[1] == 0) {
		p1.x = -w[0] / w[1];
		p1.y = 0;
		p2.x = -w[0] / w[1];
		p2.y = img.cols;
	}
	else {
		p1.x = 0;
		p1.y = -w[0] / w[2];
		p2.x = img.cols;
		p2.y = -(img.cols * w[1] + w[0]) / w[2];
	}

	line(img, p1, p2, Scalar(0, 255, 0));

	imshow("img", img);
}

std::vector<float> onlinePerceptronAlg(std::vector<std::vector<int>> X, std::vector<int> Y, int d) {
	//Initialize the variables
	std::vector<float> w = { 0.1, 0.1, 0.1 };
	float eta = pow(10, -4);
	float eLimit = pow(10, -5);
	int maxIter = pow(10, 5);
	int n = Y.size();

	for (int i = 0; i < maxIter; i++) {
		float e = 0.0f;
		for (int j = 0; j < n; j++) {
			float z = 0.0f;
			for (int k = 0; k <= d; k++) {
				z += w[k] * X[j][k];
			}
			if (z * Y[j] <= 0) {
				w[0] += eta * X[j][0] * Y[j];
				w[1] += eta * X[j][1] * Y[j];
				w[2] += eta * X[j][2] * Y[j];
				e++;
			}
		}
		e /= n;
		if (e < eLimit) {
			break;
		}
	}

	return w;
}

void onlinePerceptron() {
	char fname[MAX_PATH];

	while (openFileDlg(fname)) {
		Mat img = imread(fname, CV_LOAD_IMAGE_COLOR);
		int height = img.rows;
		int width = img.cols;

		std::vector<std::vector<int>> X;
		std::vector<int> Y;

		constructTrainingSet(img, X, Y);

		std::vector<float> w = onlinePerceptronAlg(X, Y, 2);

		//Draw the final decision boundary
		//line equation: ax + by + c = 0
		//a = w[1], b = w[2], c = w[0]
		drawLine(img, w);

		waitKey(0);
	}
}

std::vector<float> batchPerceptronAlg(std::vector<std::vector<int>> X, std::vector<int> Y, int d, Mat img) {
	//Initialize the variables
	std::vector<float> w = { 0.1, 0.1, 0.1 };
	float eta = pow(10, -4);
	float eLimit = pow(10, -5);
	int maxIter = pow(10, 5);
	int n = Y.size();

	for (int i = 0; i < maxIter; i++) {
		float E = 0.0f;
		float L = 0.0f;
		std::vector<float> dL = { 0.0f, 0.0f, 0.0f };

		for (int j = 0; j < n; j++) {
			float z = 0.0f;

			for (int k = 0; k <= d; k++) {
				z += w[k] * X[j][k];
			}

			if (z * Y[j] <= 0) {
				dL[0] -= Y[j] * X[j][0];
				dL[1] -= Y[j] * X[j][1];
				dL[2] -= Y[j] * X[j][2];
				E++;
				L -= Y[j] * z;
			}
		}

		E /= n;
		L /= n;
		dL[0] /= n;
		dL[1] /= n;
		dL[2] /= n;

		if (E < eLimit) {
			break;
		}

		w[0] -= eta * dL[0];
		w[1] -= eta * dL[1];
		w[2] -= eta * dL[2];

		drawLine(img, w);
	}

	return w;
}

void batchPerceptron() {
	char fname[MAX_PATH];

	while (openFileDlg(fname)) {
		Mat img = imread(fname, CV_LOAD_IMAGE_COLOR);
		int height = img.rows;
		int width = img.cols;

		std::vector<std::vector<int>> X;
		std::vector<int> Y;

		constructTrainingSet(img, X, Y);

		std::vector<float> w = batchPerceptronAlg(X, Y, 2, img);

		//Draw the final decision boundary
		//line equation: ax + by + c = 0
		//a = w[1], b = w[2], c = w[0]
		drawLine(img, w);

		waitKey(0);
	}
}

//Lab 12 - AdaBoost method
weakLearner findWeakLearner(std::vector<std::vector<int>> X, std::vector<int> Y, std::vector<float> w, Mat img) {
	weakLearner best_h;
	float best_err = std::numeric_limits<float>::infinity();
	
	for (int j = 1; j < X[0].size(); j++) {
		for (int th = 0; th < img.rows; th++) {
			for (int label = -1; label < 2; label += 2) {
				float e = 0;
				for (int i = 0; i < X.size(); i++) {
					int zi = 0;
					if (X[i][j] < th) {
						zi = label;
					}
					else {
						zi = -label;
					}

					if (zi * Y[i] < 0) {
						e += w[i];
					}
				}

				if (e < best_err) {
					best_err = e;
					best_h.feature_i = j - 1;
					best_h.threshold = th;
					best_h.class_label = label;
					best_h.error = e;
				}
			}
		}
	}

	return best_h;
}

classifier createClassifier(std::vector<std::vector<int>> X, std::vector<int> Y, std::vector<float> w, Mat img, int T) {
	classifier clf;
	for (int t = 0; t < T; t++) {
		weakLearner wl = findWeakLearner(X, Y, w, img);
		float alpha = 0.5 * log((1 - wl.error) / wl.error);
		float s = 0;

		for (int i = 0; i < X.size(); i++) {
			Mat row(1, 2, CV_32FC1);
			row.at<float>(0, 0) = X[i][1];
			row.at<float>(0, 1) = X[i][2];
			w[i] = w[i] * exp(-alpha * Y[i] * wl.classify(row));
			s += w[i];
		}

		//normalize the weights
		for (int i = 0; i < X.size(); i++) {
			w[i] = w[i] / s;
		}

		clf.alphas->push_back(alpha);
		clf.hs->push_back(wl);
	}
	clf.T = T;

	return clf;
}

void drawBoundary(Mat img, classifier clf) {
	Mat dst;
	img.copyTo(dst);
	cout << clf.T;

	for (int h = 0; h < img.rows; h++) {
		for (int w = 0; w < img.cols; w++) {
			Vec3b pixel = img.at<Vec3b>(h, w);
			if (pixel[0] == 255 && pixel[1] == 255 && pixel[2] == 255) {
				float sum = 0;
				Mat X(1, 2, CV_32FC1);
				X.at<float>(0, 0) = w;
				X.at<float>(0, 1) = h;

				for (int i = 0; i < clf.T; i++) {
					sum += clf.alphas->at(i) * clf.hs->at(i).classify(X);
				}

				if (sum < 0) {
					//class blue
					dst.at<Vec3b>(h, w)[0] = 204;
					dst.at<Vec3b>(h, w)[1] = 204;
					dst.at<Vec3b>(h, w)[2] = 0;
				}
				else {
					dst.at<Vec3b>(h, w)[0] = 0;
					dst.at<Vec3b>(h, w)[1] = 255;
					dst.at<Vec3b>(h, w)[2] = 255;
				}
			}
			else {
				dst.at<Vec3b>(h, w) = img.at<Vec3b>(h, w);
			}
		}
	}

	imshow("Init img", img);
	imshow("Boundary img", dst);
	waitKey(0);
}

void adaBoost() {
	char fname[MAX_PATH];

	while (openFileDlg(fname)) {
		Mat img = imread(fname, CV_LOAD_IMAGE_COLOR);
		int height = img.rows;
		int width = img.cols;

		std::vector<std::vector<int>> X;
		std::vector<int> Y;
		constructTrainingSet(img, X, Y);

		std::vector<float> w;
		int n = X.size();

		for (int i = 0; i < n; i++) {
			w.push_back(1.0f / n);
		}

		int T = 13;
		std::vector<float> alphas;

		classifier clf = createClassifier(X, Y, w, img, T);

		drawBoundary(img, clf);
	}
}

int main()
{
	int op;
	std::vector<Point_<float>> pointsFloat;
	Mat_<uchar> img;
	float* teta;
	//Lab 4
	Mat dst;

	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - Lab 1\n");
		printf(" 11 - Lab 2: RANSAC, fitting a line to a set of points\n");
		printf(" 12 - Lab 3: Rough Transform for line detection\n");
		printf(" 13 - Lab 4: Distance Transform\n");
		printf(" 14 - Lab 5: Correlation chart\n");
		printf(" 15 - Lab 5: Plot density function\n");
		printf(" 16 - Lab 7: Principal Component Analysis\n");
		printf(" 17 - Lab 8: K-means clustering for 2D points\n");
		printf(" 18 - Lab 8: K-means clustering for grayscale image\n");
		printf(" 19 - Lab 8: K-means clustering for color image\n");
		printf(" 20 - Lab 9: Naive Bayesian Classifier: Digit Recognition Application\n");
		printf(" 21 - Lab 10: K-Nearest Neighbors Classifier\n");
		printf(" 22 - Lab 11: Perceptron Algorithm\n");
		printf(" 23 - Lab 11: Batch Perceptron Algorithm\n");
		printf(" 24 - Lab 12: AdaBoost\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				pointsFloat = readData();
				img = drawPoints(pointsFloat);
				teta = computeTeta(img, pointsFloat);
				break;
			case 11:
				ransacMethod();
				break;
			case 12:
				//Mat src = imread("Images/prs_res_Hough/image_simple.bmp", CV_LOAD_IMAGE_GRAYSCALE);
				//cannyAlgorithm(src);
				computeHoughAccumulator();
				break;
			case 13:
				//distanceTransformAlg();
				compareMatchingScores();
				break;
			case 14:
				correlationChart();
				break;
			case 15:
				plotDensityFunction();
				break;
			case 16:
				pca();
				break;
			case 17:
				kMeansForPoints();
				break;
			case 18:
				kMeansGrayscale();
				break;
			case 19:
				kMeansColor();
				break;
			case 20:
				computeTrainingMatrix();
				break;
			case 21:
				confussionMatrixAndAccuracy();
				break;
			case 22:
				onlinePerceptron();
				break;
			case 23:
				batchPerceptron();
				break;
			case 24:
				adaBoost();
				break;
				
		}
	}
	while (op!=0);
	return 0;
}