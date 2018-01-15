// openCV_test.cpp : �������̨Ӧ�ó������ڵ㡣
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

bool detectface(Mat img, IplImage* *out, IplImage* *final, CascadeClassifier& cascade, double scale, double* max_stand);		//��������Ƿ����
void cutImage(IplImage* src, IplImage* dst, CvRect rect);										//��������
bool computeEyeDistance(Mat img, CascadeClassifier& cascade, float *eye_distance);				//����ͫ�׼��
bool IsInclude(Mat img, CascadeClassifier& cascade, double scale);								//����Ƿ����ĳһ��λ
bool computeImgBlur(Mat src, IplImage* *final, double* max_stand);								//����ͼƬģ���̶�
int findmax();																					//Ѱ�ҷ���Ҫ�����������������ߵ�

//���õ��ļ����������б�
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
//		cvShowImage("������׽2", frame);
//	}	
//	return 0;
//}

//�������
int main()
{

	//double std2[1000] = { 0 };										//���������ֵ
	//int *j = 0;														//img�����±�
	//IplImage* img[1000] = { NULL };									//��ŷ���Ҫ�����������飬���ڲ�������������
	IplImage* final = NULL;												//���������ͼ��
	double* max_stand = (double*)malloc(sizeof(double));				//���������ͼ��ķ���

		/*VideoCapture capture(0);
	Mat *frame = NULL;*/

	//cvNamedWindow("������׽", CV_WINDOW_AUTOSIZE);
	CvCapture* capture = cvCreateCameraCapture(0);
	//CvCapture* capture2 = cvCreateCameraCapture(0);
	IplImage* frame = NULL;
	//HANDLE showImage = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)showImageThread, capture2, 0, NULL);

	int num = 0;														//֡����
	CascadeClassifier cascade_face, cascade_left_eye, cascade_right_eye, cascade_smile, cascade_glasses;
	if (!cascade_face.load(cascade_face_path) || !cascade_left_eye.load(cascade_left_eye_path)
		|| !cascade_right_eye.load(cascade_right_eye_path) || !cascade_smile.load(cascade_smile_path)
		 || !cascade_glasses.load(cascade_glasses_path))
	{
		printf("cascade load defeat !!\n");
		getchar();
		return -1;
	}

	
	while (1)															//��������ͷ�����ÿһ֡
	{
		//��ʾ��ǰ֡

		frame = cvQueryFrame(capture);
		Mat frame_Mat = cvarrToMat(frame);
		cvShowImage("������׽", frame);
		//cvReleaseImage(&frame);
		//��ESC�˳�����
		if (cvWaitKey(40) == 27)										//cvWaitKey�Ĳ����൱�ڶ���msһ֡��������40msһ֡��1s25֡
			break;														//��ESC���˳�

		//��ǰ֡�洢·���趨
		char path[30];
		time_t now;
		time(&now);
		sprintf_s(path, "%s%lld%s%d%s", "image", now, "-", ++num, ".jpg");
		
		//�ж��Ƿ��������,image���淵�غ�ļ��úõ�����
		IplImage *image = NULL;
		if(!detectface(frame_Mat, &image, &final, cascade_face, 1, max_stand ))
		{
			printf("δ��⵽������\n");
			continue;
		}

		//�ж�˫��ͫ�׼�࣬��ͨ�׼���С�����ж���������
		Mat eye_Mat = cvarrToMat(image);
		float eye_distance = 0;
		computeEyeDistance(eye_Mat, cascade_glasses, &eye_distance);
		printf("֡ %d ��ͫ�׼��Ϊ�� %f \n", num, eye_distance);
		if (eye_distance < 70 || eye_distance > 115)						//�ж�ͫ�׼���Ƿ����Ҫ�󣬼����Ƿ�����
		{
			printf("ͫ�׼�಻���ϣ���ǰ���Ϊ�� %f \n", eye_distance);
			continue;
		}

						
		if(!IsInclude(frame_Mat, cascade_smile, 1))							//�ж��Ƿ��������
		{
			printf("ȱ�ٱ��ӣ�\n");
			continue;
		}
				
		if (image != NULL)													//�洢����ͼ��
		{
			printf("�ļ� %s ���洢\n", &path);
			cvSaveImage(path, image, 0);
		}
		cvReleaseImage(&image);
		
	}
	/*int i = findmax();
	if (i == 0) {
		printf("�����ں���ͼ��\n");
		return 0;
	}*/
	if (*max_stand != 0)
		cvSaveImage("final.jpg", final, 0);
	else
	{
		printf("��ȡʧ�ܣ�û�и�������ͼƬ����\n");
		return -1;
	}

	//��Դ�ͷ�
	cvReleaseCapture(&capture);
	//cvDestroyWindow("������׽2");
	cvDestroyWindow("������׽");
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
	Mat gray, smallImg(cvRound(img_Mat.rows / scale), cvRound(img_Mat.cols / scale), CV_8UC1);	//cvRound�Ƕ�doubleȡ��

	cvtColor(img_Mat, gray, CV_BGR2GRAY);
	resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
	equalizeHist(smallImg, smallImg);

	t = (double)cvGetTickCount();																//ʱ���
	cascade.detectMultiScale(smallImg, faces,
		1.2, 3, CV_HAAR_SCALE_IMAGE|CASCADE_FIND_BIGGEST_OBJECT|CASCADE_DO_ROUGH_SEARCH,
		Size(30, 30));																			//ԭ����������Ϊ1.3���ֵ�Ϊ1.1��������ʶ�����
	t = (double)cvGetTickCount() - t;															//���ʱ�����
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

		//�ж�ͼƬ�Ƿ�ģ�������ͼƬģ�����򷵻�false
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
{//����CvRect��src�����и�����и�������dst
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
		printf("ͫ����Ŀ��Ϊ2   \n");
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
	Laplacian(src, tmp_gray, CV_64F, 3);							//����3x3��laplacian���Ӿ���˲�
	convertScaleAbs(tmp_gray, tmp_gray);							//�鵽0~255
	meanStdDev(tmp_gray, tmp_m1, tmp_sd1);
	m1 = tmp_m1.at<double>(0, 0);									//��ֵ  
	sd1 = tmp_sd1.at<double>(0, 0);									//��׼��  
	printf("��ֵ�� %f �� ��� %f \n", m1, sd1);
	if (sd1*sd1 < 1849 && sd1*sd1 != 0)								//ֻ���²�ģ����ͼ����ֵ1764
	{
		printf("ģ��ͼ��!\n");
		return false;
	}
	if (pow(sd1, 2) > *max_stand)									//����ȫ�ַ�������ͼ��
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
//�жϸò�λ�Ƿ���ͼƬ�д���
{
	int i = 0;
	double t = 0;
	vector<Rect> faces;
	Mat gray, smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8UC1);//cvRound�Ƕ�doubleȡ��

	cvtColor(img, gray, CV_BGR2GRAY);
	resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
	equalizeHist(smallImg, smallImg);
		
	cascade.detectMultiScale(smallImg, faces,
		1.2, 3, CV_HAAR_SCALE_IMAGE,
		Size(30, 30)); //ԭ����������Ϊ1.3���ֵ�Ϊ1.1��������ʶ�����
	
	if (faces.size() != 0)	return true;
	
	return false;
}

