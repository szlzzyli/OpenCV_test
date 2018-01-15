// openCV_test.cpp : 定义控制台应用程序的入口点。
//
#include "stdafx.h"

#include <opencv2/opencv.hpp> 
#include "opencv/cv.hpp"  
#include "opencv2/objdetect/objdetect.hpp"  
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
//#include "opencv.hpp"

#include <stdio.h>
#include <iostream>
//#include <windows.h>
#include <string>
#include <time.h>
#include <math.h>

using namespace std;
using namespace cv;

bool detectface(Mat img, IplImage* *out, IplImage* *final, CascadeClassifier& cascade, double scale, double* max_stand);		//检测人脸是否存在
void cutImage(IplImage* src, IplImage* dst, CvRect rect);										//剪裁人脸
bool computeEyeDistance(Mat img, CascadeClassifier& cascade, float *eye_distance);				//计算瞳孔间距
bool IsInclude(Mat img, CascadeClassifier& cascade, double scale);								//检测是否包含某一部位
bool computeImgBlur(Mat src, IplImage* *final, double* max_stand);								//计算图片模糊程度
int findmax();																					//寻找符合要求的人像中清晰度最高的

//所用到的级联分类器列表
String cascade_face_path = "D:/develop_tools/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";
String cascade_left_eye_path = "D:/develop_tools/opencv/sources/data/haarcascades/haarcascade_lefteye_2splits.xml";
String cascade_right_eye_path = "D:/develop_tools/opencv/sources/data/haarcascades/haarcascade_righteye_2splits.xml";
String cascade_smile_path = "D:/develop_tools/opencv/sources/data/haarcascades/haarcascade_smile.xml";
String cascade_glasses_path = "D:/develop_tools/opencv/sources/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

//DWORD WINAPI showImageThread(CvCapture* p)
//{
//	while (1)
//	{
//		IplImage* frame = cvQueryFrame(p);
//		cvShowImage("人脸捕捉2", frame);
//	}	
//	return 0;
//}

//程序入口
int main()
{

	//double std2[1000] = { 0 };										//存放清晰度值
	//int *j = 0;														//img数组下标
	//IplImage* img[1000] = { NULL };									//存放符合要求人脸的数组，用于查找最清晰人像
	IplImage* final = NULL;												//存放最清晰图像
	double* max_stand = (double*)malloc(sizeof(double));				//存放最清晰图像的方差

		/*VideoCapture capture(0);
	Mat *frame = NULL;*/

	//cvNamedWindow("人脸捕捉", CV_WINDOW_AUTOSIZE);
	CvCapture* capture = cvCreateCameraCapture(0);
	//CvCapture* capture2 = cvCreateCameraCapture(0);
	IplImage* frame = NULL;
	//HANDLE showImage = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)showImageThread, capture2, 0, NULL);

	int num = 0;														//帧计数
	CascadeClassifier cascade_face, cascade_left_eye, cascade_right_eye, cascade_smile, cascade_glasses;
	if (!cascade_face.load(cascade_face_path) || !cascade_left_eye.load(cascade_left_eye_path)
		|| !cascade_right_eye.load(cascade_right_eye_path) || !cascade_smile.load(cascade_smile_path)
		 || !cascade_glasses.load(cascade_glasses_path))
	{
		printf("cascade load defeat !!\n");
		getchar();
		return -1;
	}

	
	while (1)															//遍历摄像头捕获的每一帧
	{
		//显示当前帧

		frame = cvQueryFrame(capture);
		Mat frame_Mat = cvarrToMat(frame);
		cvShowImage("人脸捕捉", frame);
		//cvReleaseImage(&frame);
		//按ESC退出遍历
		if (cvWaitKey(40) == 27)										//cvWaitKey的参数相当于多少ms一帧，现在是40ms一帧，1s25帧
			break;														//按ESC就退出

		//当前帧存储路径设定
		char path[30];
		time_t now;
		time(&now);
		sprintf_s(path, "%s%lld%s%d%s", "image", now, "-", ++num, ".jpg");
		
		//判断是否包含人脸,image保存返回后的剪裁好的人脸
		IplImage *image = NULL;
		if(!detectface(frame_Mat, &image, &final, cascade_face, 1, max_stand ))
		{
			printf("未检测到人脸！\n");
			continue;
		}

		//判断双眼瞳孔间距，若通孔间距过小，则判定不是正脸
		Mat eye_Mat = cvarrToMat(image);
		float eye_distance = 0;
		computeEyeDistance(eye_Mat, cascade_glasses, &eye_distance);
		printf("帧 %d 的瞳孔间距为： %f \n", num, eye_distance);
		if (eye_distance < 70 || eye_distance > 115)						//判断瞳孔间距是否符合要求，即脸是否正脸
		{
			printf("瞳孔间距不符合，当前间距为： %f \n", eye_distance);
			continue;
		}

						
		if(!IsInclude(frame_Mat, cascade_smile, 1))							//判断是否包含鼻子
		{
			printf("缺少鼻子！\n");
			continue;
		}
				
		if (image != NULL)													//存储合适图像
		{
			printf("文件 %s 被存储\n", &path);
			cvSaveImage(path, image, 0);
		}
		cvReleaseImage(&image);
		
	}
	/*int i = findmax();
	if (i == 0) {
		printf("不存在合适图像！\n");
		return 0;
	}*/
	if (*max_stand != 0)
		cvSaveImage("final.jpg", final, 0);
	else
	{
		printf("提取失败，没有高清晰度图片返回\n");
		return -1;
	}

	//资源释放
	cvReleaseCapture(&capture);
	//cvDestroyWindow("人脸捕捉2");
	cvDestroyWindow("人脸捕捉");
	//cvReleaseImage(&frame);
	//cvReleaseImage(&final);
	free(max_stand);
	return 0;
}

bool detectface(Mat img_Mat, IplImage* *out, IplImage* *final, CascadeClassifier& cascade, double scale, double* max_stand)
{
	int i = 0;
	double t = 0;
	vector<Rect> faces;
	Mat gray, smallImg(cvRound(img_Mat.rows / scale), cvRound(img_Mat.cols / scale), CV_8UC1);	//cvRound是对double取整

	cvtColor(img_Mat, gray, CV_BGR2GRAY);
	resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
	equalizeHist(smallImg, smallImg);

	t = (double)cvGetTickCount();																//时间戳
	cascade.detectMultiScale(smallImg, faces,
		1.2, 3, CV_HAAR_SCALE_IMAGE|CASCADE_FIND_BIGGEST_OBJECT|CASCADE_DO_ROUGH_SEARCH,
		Size(30, 30));																			//原先扩张因子为1.3，现调为1.1，可增大识别面积
	t = (double)cvGetTickCount() - t;															//检测时间计算
	//printf("detection time = %g ms\n", t / ((double)cvGetTickFrequency()*1000.));
	for (vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++)
	{
		IplImage* src = cvCloneImage(&IplImage(img_Mat));
		int r_x, r_y, r_width, r_height;
		r_x = std::max(0, r->x - 12);
		r_y = std::max(0, r->y - 20);
		r_width = std::min(r->width + 24 + r_x, src->width);
		r_height = std::min(r->height + 40 + r_y, src->height);
		r_width -= r_x;
		r_height -= r_y;

		IplImage* temp = cvCreateImage(cvSize(r_width, r_height),
			src->depth,
			src->nChannels);
		
		cutImage(src, temp, cvRect(r_x, r_y, r_width, r_height));
		try{
			cvReleaseImage(&src);
		}
		catch (cv::Exception)
		{
			printf("cv::Exception :  in cvReleaseImage(self)\n" );
		}

		//判断图片是否模糊，如果图片模糊，则返回false
		if (!computeImgBlur(cvarrToMat(temp), final, max_stand))
		{
			cvReleaseImage(&temp);
			return false;
		}
		else
		{
			printf("x = %d, y = %d, width = %d, height = %d \n", r->x, r->y, r->width, r->height);			
		}		
		*out = cvCloneImage(temp);
		//cvCopy(temp, &out);
		//out = temp;
		//cvReleaseImage(&temp);
		return true;
	}

	return false;
}

void cutImage(IplImage* src, IplImage* dst, CvRect rect)
{//按照CvRect对src进行切割，并将切割结果存入dst
	try
	{
		//cvSetImageROI(src, rect);
		if (src->roi)
		{
			src->roi->xOffset = rect.x;
			src->roi->yOffset = rect.y;
			src->roi->width = rect.width;
			src->roi->height = rect.height;
		}
		else
		{
			IplROI *roi = 0;
			roi = (IplROI*)cvAlloc(sizeof(*roi));
			roi->coi = 0;
			roi->xOffset = rect.x;
			roi->yOffset = rect.y;
			roi->width = rect.width;
			roi->height = rect.height;
			src->roi = roi;
		}

		cvCopy(src, dst, 0);

		cvResetImageROI(src);
	}
	catch (cv::Exception) {
		printf("ROI Exception!!!\n");
	}
}
                                                                                                                                                                                                                                                                                               
bool computeEyeDistance(Mat img, CascadeClassifier& cascade, float *eye_distance)
{
	vector<Rect> faces;
	Mat gray;
	//float center[2][2] = {0};
	Point2d center[2];
	cvtColor(img, gray, CV_BGR2GRAY);
	cascade.detectMultiScale(gray, faces,
		1.1, 3, 0|CV_HAAR_SCALE_IMAGE,
		Size(30, 30));
	if (faces.size() != 2)
	{
		printf("瞳孔数目不为2   \n");
		return false;
	}
	int i = 0;
	for (vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++)
	{
		center[i].x = cvRound(r->x + r->width * 0.5);
		center[i].y = cvRound(r->y + r->height * 0.5);
	}
	*eye_distance = std::sqrt(pow((center[0].x - center[1].x), 2) + pow((center[0].y - center[1].y), 2));
	return true;
}

bool computeImgBlur(Mat src, IplImage* *final, double* max_stand)
{
	Mat tmp_m1, tmp_sd1;
	double m1 = 0, sd1 = 0;
	Mat tmp_gray;
	Laplacian(src, tmp_gray, CV_64F, 3);							//利用3x3的laplacian算子卷积滤波
	convertScaleAbs(tmp_gray, tmp_gray);							//归到0~255
	meanStdDev(tmp_gray, tmp_m1, tmp_sd1);
	m1 = tmp_m1.at<double>(0, 0);									//均值  
	sd1 = tmp_sd1.at<double>(0, 0);									//标准差  
	printf("均值： %f ， 方差： %f \n", m1, sd1);
	if (sd1*sd1 < 1849 && sd1*sd1 != 0)								//只留下不模糊的图像，阈值1764
	{
		printf("模糊图像!\n");
		return false;
	}
	if (pow(sd1, 2) > *max_stand)									//保存全局方差最大的图像
	{
		*max_stand = pow(sd1, 2);
		*final = cvCloneImage(&(IplImage)src);
	}
	return true;
}

//int findmax()
//{
//	double k = std2[0];
//	int m = 0;
//	for (int i = 0; std2[i] !=0; i++)
//	{
//		if (k < std2[i])
//		{
//			k = std2[i];
//			m = i;
//		}
//	}
//	return m;
//}

bool IsInclude(Mat img, CascadeClassifier& cascade, double scale)
//判断该部位是否在图片中存在
{
	int i = 0;
	double t = 0;
	vector<Rect> faces;
	Mat gray, smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8UC1);//cvRound是对double取整

	cvtColor(img, gray, CV_BGR2GRAY);
	resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
	equalizeHist(smallImg, smallImg);
		
	cascade.detectMultiScale(smallImg, faces,
		1.2, 3, CV_HAAR_SCALE_IMAGE,
		Size(30, 30)); //原先扩张因子为1.3，现调为1.1，可增大识别面积
	
	if (faces.size() != 0)	return true;
	
	return false;
}

