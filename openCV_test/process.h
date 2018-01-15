#pragma once
//根据眼睛坐标对图像进行仿射变换
//src - 原图像
//landmarks - 原图像中68个关键点
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
	//计算两眼中心点，按照此中心点进行旋转， 第31个为左眼坐标，36为右眼坐标
	Point2f eyesCenter = Point2f((landmarks[31].x + landmarks[36].x) * 0.5f, (landmarks[31].y + landmarks[36].y) * 0.5f);

	// 计算两个眼睛间的角度
	double dy = (landmarks[36].y - landmarks[31].y);
	double dx = (landmarks[36].x - landmarks[31].x);
	double angle = atan2(dy, dx) * 180.0 / CV_PI; // Convert from radians to degrees.

												  //由eyesCenter, andle, scale按照公式计算仿射变换矩阵，此时1.0表示不进行缩放
	Mat rot_mat = getRotationMatrix2D(eyesCenter, angle, 1.0);

	Mat rot;
	// 进行仿射变换，变换后大小为src的大小
	warpAffine(src, rot, rot_mat, src.size());
	vector<Point2f> marks;

	//按照仿射变换矩阵，计算变换后各关键点在新图中所对应的位置坐标。
	for (int n = 0; n<landmarks.size(); n++)
	{
		Point2f p = Point2f(0, 0);
		p.x = rot_mat.ptr<double>(0)[0] * landmarks[n].x + rot_mat.ptr<double>(0)[1] * landmarks[n].y + rot_mat.ptr<double>(0)[2];
		p.y = rot_mat.ptr<double>(1)[0] * landmarks[n].x + rot_mat.ptr<double>(1)[1] * landmarks[n].y + rot_mat.ptr<double>(1)[2];
		marks.push_back(p);
	}
	//标出关键点
	for (int j = 0; j < landmarks.size(); j++)
	{
		circle(rot, marks[j], 2, Scalar(0, 0, 255));
	}
	return rot;
}