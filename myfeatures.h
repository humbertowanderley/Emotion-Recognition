#ifndef FEATURES_INCLUDED
#define FEATURES_INCLUDED
using namespace dlib;
#define NUM_FEATURES 9
#define NUM_CLASSES 6
double length(point a,point b);
point center(full_object_detection shape);

double openessMouth(full_object_detection shape);
double widthEye(full_object_detection shape);
double widthMouth(full_object_detection shape);

double nariz_sobrancelha1Mid(full_object_detection shape);
double nariz_sobrancelha1Mid2(full_object_detection shape);
double alturaBoca1(full_object_detection shape);
double alturaBoca2(full_object_detection shape);
double aberturaOlho1(full_object_detection shape);
double aberturaOlho2(full_object_detection shape);

//double heigthEyebrow1(full_object_detection shape);
//double heigthEyebrow2(full_object_detection shape);
//double tipLip_nose(full_object_detection shape); //dist. do nariz à ponta da boca
std::vector<double> featuresExtraction(std::vector<full_object_detection> shapes);
#include "myfeatures.cpp"
#endif // FEATURES_INCLUDED



