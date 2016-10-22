#ifndef FEATURES_INCLUDED
#define FEATURES_INCLUDED
using namespace dlib;
#define NUM_FEATURES 8
#define NUM_CLASSES 5
double length(point a,point b);
point center(full_object_detection shape);
double openessMouth(full_object_detection shape);
double widthEye(full_object_detection shape);
double widthMouth(full_object_detection shape);
double heigthEyebrow1(full_object_detection shape);
double heigthEyebrow2(full_object_detection shape);
double tipLip_nose(full_object_detection shape); //dist. do nariz à ponta da boca
std::vector<double> featuresExtraction(std::vector<full_object_detection> shapes);
#include "myfeatures.cpp"
#endif // FEATURES_INCLUDED



