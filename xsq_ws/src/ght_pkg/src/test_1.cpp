// 28行 更改文件路径 

#include <stdlib.h>
#include <iostream>
#include <string>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Bool.h"
#include "std_msgs/Float32.h"
#include <geometry_msgs/Twist.h>
#define PI 3.1415

using namespace std;
using namespace cv;

// 图像尺寸
const int kRowNumLowResImg = 200;           //纵坐标最大值
const int kColNumLowResImg = 200;           //横坐标最大值

// 文件路径
const string kFilenameTemplate = "/home/zdh/xsq_ws/img/template/template.jpg";

Mat HsvSegment(Mat img_origin)
{
	Mat img_hsv;
	cvtColor(img_origin, img_hsv, COLOR_BGR2HSV);
	// Green
	// inRange(img_hsv, Scalar(35, 43, 46), Scalar(77, 255, 255), img_hsv);
	// Black
	inRange(img_hsv, Scalar(0, 0, 0), Scalar(180, 255, 46), img_hsv);
	return img_hsv;
}

Mat PreprocessTemplate(Mat img_origin)
{
	//功能：广义霍夫变换模板预处理
	//输入：模板
	//输出：预处理后的模板
	//操作：低分辨率处理；彩色转灰度；高斯滤波；Canny 边沿检测

	const float kGaussianKernSigma = 1.5;      //高斯卷积核方差
	const float kGaussianKernSize = 5;         //高斯卷积核尺寸
	const float kCannyThresh1 = 70;            //Canny双阈值处理
	const float kCannyThresh2 = 150;           //Canny双阈值处理

	Mat img_low_res;
	resize(img_origin,img_low_res,cv::Size(kColNumLowResImg, kRowNumLowResImg),CV_INTER_AREA);   
	Mat img_gray;
	cvtColor(img_low_res, img_gray, CV_BGR2GRAY);                                    
	Mat img_gaussian_blur;
	GaussianBlur(img_gray, img_gaussian_blur, cv::Size(kGaussianKernSize,kGaussianKernSize), kGaussianKernSigma);
	Mat img_canny;
	Canny(img_gaussian_blur, img_canny, kCannyThresh1, kCannyThresh2, 3, false);
	imshow("Canny",img_canny);
	return img_canny;
}

Mat PreprocessImg(Mat img_origin)
{
	/* 功能：广义霍夫变换预处理
	 * 输入：原图像
	 * 输出：预处理后的图像
	 * 操作：低分辨率处理；高斯滤波；Canny 边沿检测
	 */
	const float kGaussianKernSigma = 1.5;      //高斯卷积核方差
	const float kGaussianKernSize = 5;         //高斯卷积核尺寸
	const float kCannyThresh1 = 70;            //Canny双阈值处理
	const float kCannyThresh2 = 150;           //Canny双阈值处理

	Mat img_low_res;
	resize(img_origin,img_low_res,cv::Size(kColNumLowResImg, kRowNumLowResImg),CV_INTER_AREA);   
                                
	Mat img_gaussian_blur;
	GaussianBlur(img_low_res, img_gaussian_blur, cv::Size(kGaussianKernSize,kGaussianKernSize), kGaussianKernSigma);
	Mat img_canny;
	Canny(img_gaussian_blur, img_canny, kCannyThresh1, kCannyThresh2, 3, false);
	imshow("Canny",img_canny);
	return img_canny;
}

Mat InversePerspectiveMapping(const int (*src_points)[2], const int (*dst_points)[2])
{
	/* 功能：单次逆透射变换完成相机的几何畸变校正
	 * 输入：参考四边形顶点坐标（二维数组行指针）；变换后矩形四顶点坐标（二维数组行指针）
	 * 输出：单应矩阵
	 */

    vector<Point2f> p,q;
    p.push_back(Point2f(src_points[0][0],src_points[0][1]));
    q.push_back(Point2f(dst_points[0][0],dst_points[0][1]));
	
	p.push_back(Point2f(src_points[1][0],src_points[1][1]));
    q.push_back(Point2f(dst_points[1][0],dst_points[1][1]));

    p.push_back(Point2f(src_points[2][0],src_points[2][1]));
	q.push_back(Point2f(dst_points[2][0],dst_points[2][1]));


    p.push_back(Point2f(src_points[3][0],src_points[3][1]));
	q.push_back(Point2f(dst_points[3][0],dst_points[3][1]));

	Mat matrix_perspective_transform = getPerspectiveTransform(p,q);
	
	return matrix_perspective_transform; 
}

int ConstructRTable(Mat img_origin, float (*r_table)[3])
{
	/* 功能：建立模板的r_table
	 * 输入：模板图片；r_table二维数组的行指针
	 * 输出：r_table的行数
	 */

	// 广义霍夫变换之r-table的采样率与阈值
	const float kAngleSamplingNum = 50;       //角采样率
	const int kThreshTemplateGrayScale = 100;  //灰度阈值

    // Preprocess
    Mat img_after_preprocessing;			
    img_after_preprocessing = PreprocessTemplate(img_origin);
    // Gradient
    Mat grad_x, grad_y, grad_mag, grad_ang;				    //x方向梯度, y方向梯度, 梯度模长, 梯度角度
    Sobel(img_after_preprocessing, grad_x, CV_32FC1, 1, 0); //x方向梯度
    Sobel(img_after_preprocessing, grad_y, CV_32FC1, 0, 1); //y方向梯度
    magnitude(grad_x, grad_y, grad_mag);          //梯度幅值
    phase(grad_x, grad_y, grad_ang, false);       //梯度角（使用弧度制，若true则为角度制
    // xy-table
	int row_num_r_table = 0;		
	float xy_table[kRowNumLowResImg*kColNumLowResImg][3];
	int x,y = 0;                                  //循环变量，作为横纵坐标
	int y_reference_point = ceil(0.5 * kRowNumLowResImg); //模板参考点y坐标（中心点）		
	int x_reference_point = ceil(0.5 * kColNumLowResImg); //模板参考点x坐标（中心点）
	for (int i = 0; i < kAngleSamplingNum; i++)   //遍历所有角度			
	{
		for (int x = 0; x < kColNumLowResImg; x++)//遍历x坐标
		{
			// 求得该直线上对应y坐标的位置
			y = ceil(tan(i*(2*PI)/kAngleSamplingNum)*(x-x_reference_point)+y_reference_point);		
			// 直线不能越出图片边界，忽略强度较小的边沿
			if (y > 0 && y < kRowNumLowResImg && img_after_preprocessing.at<uchar>(y,x) > kThreshTemplateGrayScale) 
			{
				xy_table[row_num_r_table][0] = i*(2*PI)/kAngleSamplingNum;
				xy_table[row_num_r_table][1] = x;
				xy_table[row_num_r_table][2] = y;
				row_num_r_table ++;
			}
		}
	}

	row_num_r_table++;//得到r-table的行数（模板精细程度的指标，基本由“图像基础尺寸”与“角采样数”决定）
	// r-table
	for (int i = 0; i < row_num_r_table; i++)
	{
		r_table[i][0] = grad_ang.at<float>(xy_table[i][2],xy_table[i][1]);
		r_table[i][1] = sqrt(pow((xy_table[i][1]-x_reference_point),2) + pow((xy_table[i][2]-y_reference_point),2) );
		r_table[i][2] = xy_table[i][0]+PI;
	}

	return row_num_r_table;
}

int GeneralHoughTransform(Mat img_origin, float (*r_table)[3], int row_num_r_table, int& x_target, int& y_target)
{
	/* 功能：利用广义霍夫变换完成物体识别
	 * 输入：待识别图片；模板的r-table；r_table数组的行数；目标x坐标的地址；目标y坐标的地址
	 * 输出：最高票数
	 */	

	// 广义霍夫变换之模板匹配
	const float kThreshGradAngleDiff = 0.01;   //模板与目标图像梯度角的误差最大值
	const float kMinZoomFactor = 0.3;          //最小缩放倍数          
	const float kMaxZoomFactor = 1.2;          //最大缩放倍数
	const int kScaleSamplingNum = 3;           //尺寸采样数
	const int kRotationSamplingNum = 2;        //旋转采样数
	const int kRotationSymmetry = 2;           //H旋转180°后与自身重合，利用此对称性减少不必要的计算
	// Preprocess
 	Mat img_after_preprocessing;						  
	img_after_preprocessing = PreprocessImg(img_origin);
	// Gradient
	Mat grad_x, grad_y, grad_mag, grad_ang;				  //x方向梯度, y方向梯度, 梯度模长, 梯度角度
	Sobel(img_after_preprocessing, grad_x, CV_32FC1, 1, 0); // x方向梯度
	Sobel(img_after_preprocessing, grad_y, CV_32FC1, 0, 1); // y方向梯度
	magnitude(grad_x, grad_y, grad_mag);     //梯度幅值
	phase(grad_x, grad_y, grad_ang, false);  //梯度角（使用弧度制，若true则为角度制
	// Voting
   	int idx_scale, idx_rotation = 0;//循环变量，遍历模板的尺度与旋转
	float zoom_factor, angle = 0;//对应的缩放倍数与旋转角度
	int i,j = 0;//选民的横纵坐标
   	int x,y = 0;//候选人的横、纵坐标		
	int counter[kScaleSamplingNum][kRotationSamplingNum][kRowNumLowResImg][kColNumLowResImg] = {0};// 计数器
	int maximum_counter = 0;							  //最大计数	
	float zoom_factor_maximum_counter = 0;
	float angle_maximum_counter = 0;
	int x_maximum_counter = 0;                            //低分辨率图像中目标的横坐标
	int y_maximum_counter = 0;                            //低分辨率图像中目标的纵坐标

	for (int idx_scale = 0; idx_scale < kScaleSamplingNum; idx_scale++)
	{
		zoom_factor = kMinZoomFactor + idx_scale*(kMaxZoomFactor - kMinZoomFactor)/kScaleSamplingNum;
		for (int idx_rotation = 0; idx_rotation < kRotationSamplingNum; idx_rotation++)
		{
			angle = idx_rotation*2*PI/kRotationSymmetry/kRotationSamplingNum; // H is symmetric, so the coefficient of PI is one.
			for (int j = 0; j < kRowNumLowResImg; j++)
			{
				for (int i = 0; i < kColNumLowResImg; i++)
				{
					for (int k = 0; k < row_num_r_table; k++)
					{
						//判断(i,j)像素是否有投票资格
						//1. (i,j)像素的灰度值不为0（预筛选，提速）
						//2. (i,j)像素处的梯度角在r-table名单第一列中出现过（细筛选）
						if (img_after_preprocessing.at<uchar>(j,i) != 0 && abs(grad_ang.at<float>(j,i) - r_table[k][0]) < kThreshGradAngleDiff)
						{
							//计算(i,j)像素的投票对象(x,y)
							x = ceil(i+r_table[k][1]*zoom_factor*cos(r_table[k][2]+angle));
							y = ceil(j+r_table[k][1]*zoom_factor*sin(r_table[k][2]+angle));
							//确保(x,y)不能越界
							if (x >= 0 && x < kColNumLowResImg && y >= 0 && y < kRowNumLowResImg)
							{
								counter[idx_scale][idx_rotation][y][x]++;
							}
						}
					}
				}
			}
		}
	}
    
	// Sifting

    // 利用 std::max_element 找到 counter 最大值 
	maximum_counter = *std::max_element(&counter[0][0][0][0], &counter[0][0][0][0]+kRowNumLowResImg*kColNumLowResImg*kScaleSamplingNum*kRotationSamplingNum);    
    // 四层循环找到最大值
    bool flag = false;
    for (int idx_scale = 0; idx_scale < kScaleSamplingNum && flag == false; idx_scale++)
    {
    	for (int idx_rotation = 0; idx_rotation < kRotationSamplingNum && flag == false; idx_rotation++)
    	{
    		for (int y = 0; y < kRowNumLowResImg && flag == false; y++)
    		{
    			for (int x = 0; x < kColNumLowResImg && flag == false; x++)
    			{
    				if (counter[idx_scale][idx_rotation][y][x] == maximum_counter)
    				{
    					zoom_factor_maximum_counter = kMinZoomFactor + idx_scale*(kMaxZoomFactor - kMinZoomFactor)/kScaleSamplingNum;
    					angle_maximum_counter = 360/kRotationSymmetry*idx_rotation/kRotationSamplingNum;
    					y_maximum_counter = y;
    					x_maximum_counter = x;
    					flag = true;
    				}
    			}
    		}
    	}
    }

    // 得到原图中的目标点位置
	x_target = round(x_maximum_counter*img_origin.size().width/kColNumLowResImg);
    y_target = round(y_maximum_counter*img_origin.size().height/kRowNumLowResImg);
	return maximum_counter; 
}


Mat Mark(Mat img_origin, int x_target, int y_target)
{
	/* 功能：在给定像素位置标记彩色方块，便于展示
	 * 输入：待标记图片；目标横坐标的值；目标纵坐标的值
	 * 输出：标记后的图片
	 */

	const int kSizeDisplayBlock = 10;          // 输出图像上目标点处的正方形像素块的大小

 	Mat img_target_display = img_origin.clone();
	//越界检查
	if (x_target > kSizeDisplayBlock 
		&& x_target + kSizeDisplayBlock < img_target_display.size().width 
		&& y_target > kSizeDisplayBlock 
		&& y_target + kSizeDisplayBlock < img_target_display.size().height)
	{
	//	cout<<"0"<<endl;
		for (int j = y_target - kSizeDisplayBlock; j < y_target + kSizeDisplayBlock; j++)
		{
	//		cout<<"1"<<endl;
			for (int i = x_target - kSizeDisplayBlock; i < x_target + kSizeDisplayBlock; i++)				
			{
	//			cout<<"2"<<endl;				
				img_target_display.at<Vec3b>(j,i)[0] = 255;
				img_target_display.at<Vec3b>(j,i)[1] = 0;
				img_target_display.at<Vec3b>(j,i)[2] = 0;
			}
		}
	}
	else 
	{
		cout<<"Target is on the edge of the detection zone."<<endl;
	}
	return img_target_display;
}

void Navigate(int x_target, int y_target, int cols, int rows, int poll, bool& flag, float& linear_vel, float& angular_vel)
{
	// 速度范围
	const float kMinAngularVel = 0.1;
	const float kMaxAngularVel = 0.25;
	const float kMinLinearVel = 0.1;
	const float kMaxLinearVel = 0.3;
	// 线速度开关控制参数
	const int kPollThresh1 = 10; // 轻视票数低于此阈值的目标
	const int kPollThresh2 = 30;// 重视票数高于此阈值的目标
	// 角速度P控制参数
	const float kProportion = 0.0020;
	// 中轴像素点范围
	const int kCenterRange = 200;

	int error = cols/2-x_target;	// 偏差

	// 角速度：P控制，偏差为目标到画面中轴线的距离
	
	angular_vel = kProportion*error;
	if (abs(angular_vel) > kMaxAngularVel)
	{
		if(angular_vel >= 0)    angular_vel = kMaxAngularVel;
		if(angular_vel < 0)    angular_vel = -kMaxAngularVel;
	}
	if (abs(angular_vel) < kMinAngularVel)
	{
		if(angular_vel >= 0)    angular_vel = kMinAngularVel;
		if(angular_vel < 0)    angular_vel = -kMinAngularVel;
	}

	// 线速度：开关控制
	if ( abs(error) <= kCenterRange )
	{
		if (poll > kPollThresh1 && poll < kPollThresh2)
		{
			linear_vel = kMinAngularVel;
		}
		if (poll >= kPollThresh2)
		{
			linear_vel = kMaxLinearVel;
		}
	}
	// 停车条件：
	if (rows - y_target < 0.1*rows && poll > kPollThresh1)
	{
		flag = 1;
		angular_vel = 0;
		linear_vel = 0.1;
	} 
}


int main(int argc, char **argv)
{

	// const int kCamera = 0; 
	// 几何畸变校正的参考四边形坐标，(x,y)左上、右上、左下、右下
	const int kQuadrangle[4][2] = {
	{161,231},
	{496,234},
	{0,375},
	{670,375}};
	// 映射后的矩形坐标
	const int kFrame[4][2] = {
	{0,0},
	{670,0},
	{0,370},
	{670,370}};
	// 初始化
	Mat img_template, img_origin, m_ipm, img_ipm, img_ipm_target_display;
	float r_table[kRowNumLowResImg*kColNumLowResImg][3] = {0};
	int maximum_counter, x_target_ipm, y_target_ipm = 0;
	// 读取模板图片
	img_template = imread(kFilenameTemplate);
	// 建立模板的r_table
	int row_num_r_table = ConstructRTable(img_template, r_table);
	// 得到单应矩阵
	m_ipm = InversePerspectiveMapping(kQuadrangle,kFrame);
	// cout << "matrix_perspective_transform:  "<<endl << format(m_ipm, Formatter::FMT_C) << endl << endl;	

	// 初始化ROS节点
	ROS_WARN("**********START**********");
	ros::init(argc,argv,"self_parking"); 
	ros::NodeHandle self_parking;        // 节点句柄
	ros::Publisher pub=self_parking.advertise<geometry_msgs::Twist>("/smoother_cmd_vel", 5);
	// 打开摄像头
	VideoCapture capture;
	capture.open(1); // 打开Dashbot摄像头为1，打开笔记本摄像头为0
	waitKey(100);
	if(!capture.isOpened())	// 摄像头异常处理
	{
		printf("摄像头图像读取失败。\n");
	 	return 0;
	}
	
	// 控制量
	float linear_vel = 0;
	float angular_vel = 0;
	// 当flag == 1时，认为已经来到了目标面前
	bool flag = 0;

	// 主循环
	while(ros::ok())
	{
		// 截取双目摄像头的一目
		capture.read(img_origin);
		img_origin = img_origin(Range(0,img_origin.rows),Range(0,img_origin.cols/2-170));
		imshow("src",img_origin);			
		// 颜色分割
		Mat img_hsv = HsvSegment(img_origin);
		imshow("img_hsv",img_hsv);
		// 逆透射变换
		warpPerspective(img_hsv, img_ipm, m_ipm, cv::Size(img_origin.cols,img_origin.rows));
		warpPerspective(img_origin, img_ipm_target_display, m_ipm, cv::Size(img_origin.cols,img_origin.rows));
		// 广义霍夫变换
		maximum_counter = GeneralHoughTransform(img_ipm, r_table, row_num_r_table, x_target_ipm, y_target_ipm);
		cout<<"maximum_counter = "<<maximum_counter<<endl;		
		// 展示识别结果
	 	img_ipm_target_display = Mark(img_ipm_target_display, x_target_ipm, y_target_ipm);	
		imshow("Target Recognition: After IPM", img_ipm_target_display);	
		// 导航
		Navigate(x_target_ipm, y_target_ipm, img_ipm.cols, img_ipm.rows, maximum_counter, flag, linear_vel, angular_vel);
		cout<<"linear_vel = "<<linear_vel<<"    "<<"angular_vel"<<"    "<<angular_vel<<endl; 
		// 发布速度信息
		ros::Publisher pub=self_parking.advertise<geometry_msgs::Twist>("/smoother_cmd_vel", 5);
		geometry_msgs::Twist cmd_navigate;
		cmd_navigate.linear.x = linear_vel;
		cmd_navigate.linear.y = 0;
		cmd_navigate.linear.z = 0;
		cmd_navigate.angular.x = 0;
		cmd_navigate.angular.y = 0;
		cmd_navigate.angular.z = angular_vel;
		pub.publish(cmd_navigate);
		// 停车
		if (flag == 1)
		{
			for (int i = 0; i<200000; i++)
			{
				pub.publish(cmd_navigate);
			}
			break;
		}
		
		ros::spinOnce();
		waitKey(5);
	}	
	return 0;
}
