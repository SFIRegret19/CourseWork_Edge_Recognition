#include <opencv2/opencv.hpp>   
#include <opencv2/imgproc.hpp>  
#include <opencv2/highgui.hpp>  
#include <opencv2/imgcodecs.hpp>
#include <iostream>             
#include <vector>               
#include <string>               
#include <algorithm>            

using namespace std; 

cv::Mat getScaledDisplayImage(const cv::Mat& sourceImage, int targetWidth, int targetHeight) {
    if (sourceImage.empty() || targetWidth <= 0 || targetHeight <= 0) {
        return sourceImage.empty() ? cv::Mat() : sourceImage.clone(); 
    }

    double originalWidth = static_cast<double>(sourceImage.cols);
    double originalHeight = static_cast<double>(sourceImage.rows);

    if (originalWidth == 0 || originalHeight == 0) {
        return sourceImage.clone();
    }

    double scaleW = static_cast<double>(targetWidth) / originalWidth;
    double scaleH = static_cast<double>(targetHeight) / originalHeight;

    double scaleFactor = std::min(scaleW, scaleH);


    cv::Mat displayImage;
    int interpolationMethod = cv::INTER_LINEAR; 

    if (sourceImage.type() == CV_8UC1) {
        bool allPixelsZeroOrMax = true; 
        for (int r = 0; r < sourceImage.rows && allPixelsZeroOrMax; ++r) {
            for (int c = 0; c < sourceImage.cols; ++c) {
                uchar val = sourceImage.at<uchar>(r, c);
                if (val != 0 && val != 255) {
                    allPixelsZeroOrMax = false;
                    break;
                }
            }
        }
        if (allPixelsZeroOrMax) {
            interpolationMethod = cv::INTER_NEAREST;
        }
    }

    cv::resize(sourceImage, displayImage, cv::Size(), scaleFactor, scaleFactor, interpolationMethod);
    return displayImage;
}

string getShapeType(const vector<cv::Point>& approx) {
    string shapeType = "Unknown";       
    size_t vertices = approx.size();    
    double area = cv::contourArea(approx); 

    if (vertices == 3) {
        shapeType = "Triangle";
    }
    else if (vertices == 4) {
        cv::Rect boundingBox = cv::boundingRect(approx); 
        float aspectRatio = 0;                           
        if (boundingBox.height > 0) {                    
            aspectRatio = (float)boundingBox.width / boundingBox.height;
        }

        if (aspectRatio >= 0.95 && aspectRatio <= 1.05) { 
            shapeType = "Square";
        }
        else {
            shapeType = "Rectangle";
        }
    }
    else if (vertices > 4) { 
        bool isCircle = false;
        double perimeter_shape = cv::arcLength(approx, true);

        if (perimeter_shape > 0 && area > 0) { 
            double circularity = (4 * CV_PI * area) / (perimeter_shape * perimeter_shape); 
            if (circularity > 0.85 && circularity < 1.15) { 
                isCircle = true;
            }
        }

        if (!isCircle) {
            cv::Point2f center_mc;
            float radius_mc;
            cv::minEnclosingCircle(approx, center_mc, radius_mc);
            if (radius_mc > 0) { 
                double circleArea_mc = CV_PI * radius_mc * radius_mc; 
                if (circleArea_mc > 0 && std::abs(1.0 - (area / circleArea_mc)) < 0.15) { 
                    isCircle = true;
                }
            }
        }

        if (isCircle) {
            shapeType = "Circle";
        }
        else { 
            if (vertices == 5) shapeType = "Pentagon";
            else if (vertices == 6) shapeType = "Hexagon";
            else shapeType = "Polygon"; 
        }
    }
    return shapeType;
}

int main() {
    const int TARGET_DISPLAY_WIDTH = 800;
    const int TARGET_DISPLAY_HEIGHT = 785;

    //string imagePath = "D:\\ОБУЧЕНИЕ ТГУ\\Курсач\\ShapeRecognitionCoursework\\x64\\Debug\\images\\simple_shapes.png"; 
    string imagePath = "D:\\ОБУЧЕНИЕ ТГУ\\Курсач\\ShapeRecognitionCoursework\\x64\\Debug\\images\\test1.tif";
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE); 

    if (image.empty()) { 
        cerr << "Error: Could not open or find the image at: " << imagePath << endl;
        return -1;
    }

    cv::imshow("1. Original Grayscale Image", getScaledDisplayImage(image, TARGET_DISPLAY_WIDTH, TARGET_DISPLAY_HEIGHT));

    cv::Mat resultImage; 
    cv::cvtColor(image, resultImage, cv::COLOR_GRAY2BGR); 


    cv::Mat blurredImage;
    cv::medianBlur(image, blurredImage, 7); 

    cv::imshow("2. Median Blurred", getScaledDisplayImage(blurredImage, TARGET_DISPLAY_WIDTH, TARGET_DISPLAY_HEIGHT));

    cv::Mat edges;      
    int threshold1 = 20; 
    int threshold2 = 60; 
    cv::Canny(blurredImage, edges, threshold1, threshold2); 

    cv::imshow("3a. Canny Edges (Raw)", getScaledDisplayImage(edges, TARGET_DISPLAY_WIDTH, TARGET_DISPLAY_HEIGHT));

    cv::Mat dilate_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)); 
    cv::dilate(edges, edges, dilate_kernel, cv::Point(-1, -1), 1);                   

    cv::imshow("3b. Edges (Dilated)", getScaledDisplayImage(edges, TARGET_DISPLAY_WIDTH, TARGET_DISPLAY_HEIGHT));

    cv::Mat kernel_open = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));  
    cv::morphologyEx(edges, edges, cv::MORPH_OPEN, kernel_open, cv::Point(-1, -1), 1);

    cv::imshow("3c. Edges (After Dilate+Open)", getScaledDisplayImage(edges, TARGET_DISPLAY_WIDTH, TARGET_DISPLAY_HEIGHT));


    vector<vector<cv::Point>> contours; 
    vector<cv::Vec4i> hierarchy;     
    cv::findContours(edges, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cout << "Found " << contours.size() << " contours (after morphology)." << endl;

    for (size_t i = 0; i < contours.size(); i++) { 
        double area = cv::contourArea(contours[i]);
        if (area < 500) { 
            continue;     
        }

        vector<cv::Point> approxPoly; 
        double perimeter = cv::arcLength(contours[i], true);
        if (perimeter == 0) continue; 
        cv::approxPolyDP(contours[i], approxPoly, 0.01 * perimeter, true);

        string shape = getShapeType(approxPoly); 
        cout << "Contour #" << i << ": vertices = " << approxPoly.size() << ", shape = " << shape << ", area = " << area << endl;

        cv::drawContours(resultImage, contours, (int)i, cv::Scalar(0, 255, 0), 4);

        cv::Moments M = cv::moments(contours[i]);
        if (M.m00 != 0) { 
            int cX = static_cast<int>(M.m10 / M.m00); 
            int cY = static_cast<int>(M.m01 / M.m00); 
            cv::putText(resultImage, shape, cv::Point(cX - 20, cY), cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(0, 0, 255), 5); 
        }
        else {
            cv::Rect br = cv::boundingRect(contours[i]); 
            cv::putText(resultImage, shape, cv::Point(br.x + br.width / 2 - 20, br.y + br.height / 2), cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(0, 0, 255), 5);
        }
    }

    cv::imshow("4. Detected Shapes", getScaledDisplayImage(resultImage, TARGET_DISPLAY_WIDTH, TARGET_DISPLAY_HEIGHT));

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}