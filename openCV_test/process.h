#pragma once
//�����۾������ͼ����з���任
//src - ԭͼ��
//landmarks - ԭͼ����68���ؼ���
#include <opencv2/opencv.hpp> 
#include "opencv/cv.hpp"  
#include "opencv2/objdetect/objdetect.hpp"  
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  

#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

Mat getwarpAffineImg(Mat &src, vector<Point2f> &landmarks)
{
	Mat oral; src.copyTo(oral);
	for (int j = 0; j < landmarks.size(); j++)
	{
		circle(oral, landmarks[j], 2, Scalar(255, 0, 0));
	}
	//�����������ĵ㣬���մ����ĵ������ת�� ��31��Ϊ�������꣬36Ϊ��������
	Point2f eyesCenter = Point2f((landmarks[31].x + landmarks[36].x) * 0.5f, (landmarks[31].y + landmarks[36].y) * 0.5f);

	// ���������۾���ĽǶ�
	double dy = (landmarks[36].y - landmarks[31].y);
	double dx = (landmarks[36].x - landmarks[31].x);
	double angle = atan2(dy, dx) * 180.0 / CV_PI; // Convert from radians to degrees.

												  //��eyesCenter, andle, scale���չ�ʽ�������任���󣬴�ʱ1.0��ʾ����������
	Mat rot_mat = getRotationMatrix2D(eyesCenter, angle, 1.0);

	Mat rot;
	// ���з���任���任���СΪsrc�Ĵ�С
	warpAffine(src, rot, rot_mat, src.size());
	vector<Point2f> marks;

	//���շ���任���󣬼���任����ؼ�������ͼ������Ӧ��λ�����ꡣ
	for (int n = 0; n<landmarks.size(); n++)
	{
		Point2f p = Point2f(0, 0);
		p.x = rot_mat.ptr<double>(0)[0] * landmarks[n].x + rot_mat.ptr<double>(0)[1] * landmarks[n].y + rot_mat.ptr<double>(0)[2];
		p.y = rot_mat.ptr<double>(1)[0] * landmarks[n].x + rot_mat.ptr<double>(1)[1] * landmarks[n].y + rot_mat.ptr<double>(1)[2];
		marks.push_back(p);
	}
	//����ؼ���
	for (int j = 0; j < landmarks.size(); j++)
	{
		circle(rot, marks[j], 2, Scalar(0, 0, 255));
	}
	return rot;
}