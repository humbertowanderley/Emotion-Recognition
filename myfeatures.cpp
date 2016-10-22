#include <dlib/gui_widgets.h>
#include <iostream>
using namespace std;

double length(point a,point b)
{
    return sqrt( (a.x()-b.x())*(a.x()-b.x()) + (a.y()-b.y())*(a.y()-b.y()) );
}

point center(full_object_detection shape)
{
    point c;
    double sumx=0,sumy=0;
    for (int i = 0; i < shape.num_parts(); ++i){
        sumx+=shape.part(i).x();
        sumy+=shape.part(i).y();
    }
    c = point(sumx/shape.num_parts(),sumy/shape.num_parts()) ;
    return c;
}

std::vector<double> normalize(std::vector<double> v)
{
    if (v.size() > 0){
        double maxv = *max_element(v.begin(), v.end()-1);
        double minv = *min_element(v.begin(), v.end()-1);
        for (int i = 0; i < v.size(); ++i){
            if (v[i]>=0){
                v[i] = (v[i]-minv)/(maxv-minv);
            }

        }
    }

    return v;
}
std::vector<double> normalize2(std::vector<double> v)
{
    double norm = 0;
    if (v.size() > 0){
        double maxv = *max_element(v.begin(), v.end()-1);
        double minv = *min_element(v.begin(), v.end()-1);
        for (int i = 0; i < v.size(); ++i){
            norm += v[i]*v[i];
        }
        norm = sqrt(norm);

        for (int i = 0; i < v.size(); ++i){
            v[i] = v[i]/norm;
        }

    }

    return v;
}

//Features

// double nariz_sobrancelha1(full_object_detection shape)    //33 - happy
// {
//     return length(shape.part(29),shape.part(22));
// }

double nariz_sobrancelha1Mid(full_object_detection shape)    //33 - happy
{
    return length(shape.part(29),shape.part(23));
}

// double nariz_sobrancelha2(full_object_detection shape)    //33 - happy
// {
//     return length(shape.part(29),shape.part(21));
// }
double nariz_sobrancelha2Mid(full_object_detection shape)    //33 - happy
{
    return length(shape.part(29),shape.part(20));
}

double alturaBoca1(full_object_detection shape)      //
{
    return length(shape.part(29),shape.part(54));
}

double alturaBoca2(full_object_detection shape)      //
{
    return length(shape.part(29),shape.part(48));
}

double aberturaOlho1(full_object_detection shape)      //
{
    return length(shape.part(47),shape.part(23));
}

double aberturaOlho2(full_object_detection shape)      //
{
    return length(shape.part(40),shape.part(20));
}










//***************************************************************************************
double openessMouth(full_object_detection shape)    //33 - abertura da boca
{
    return length(shape.part(51),shape.part(57));
}

double widthMouth(full_object_detection shape)      //largura boca // happy
{
    return length(shape.part(48),shape.part(54));
}

// double heigthMouth1(full_object_detection shape)      //sai
// {
//     return length(center(shape),shape.part(54));
// }

// double heigthMouth2(full_object_detection shape)      //sai
// {
//     return length(center(shape),shape.part(48));
// }

double widthEye(full_object_detection shape)      //
{
    return (length(shape.part(36),shape.part(39))+ length(shape.part(42),shape.part(45)))/2.0;
}

// double heigthEyebrow1(full_object_detection shape)      //
// {
//     return length(center(shape),shape.part(19));
// }

// double heigthEyebrow2(full_object_detection shape)      //
// {
//     return length(center(shape),shape.part(24));
// }
// double heigthEyebrow1(full_object_detection shape)      //
// {
//     return length(shape.part(30),shape.part(17))/2.0 + length(shape.part(30),\
//         shape.part(26))/2.0;
// }

// double heigthEyebrow2(full_object_detection shape)      //
// {
//     return length(shape.part(30),shape.part(48))/2.0 + length(shape.part(30),\
//         shape.part(54))/2.0;
// }

// double tipLip_nose(full_object_detection shape) //dist. do nariz à ponta da boca
// {
//     return length(shape.part(30),shape.part(21))/2.0 + length(shape.part(30),\
//         shape.part(22))/2.0;
// }

std::vector<double> featuresExtraction(std::vector<full_object_detection> shapes)
{
    std::vector<double> features;
    for (int i = 0; i < shapes.size(); ++i){
        // features.push_back(openessMouth(shapes[i]));
        // features.push_back(widthMouth(shapes[i]));
        // features.push_back(widthEye(shapes[i]));
        // features.push_back(heigthEyebrow1(shapes[i]));
        // features.push_back(heigthEyebrow2(shapes[i]));
        // features.push_back(heigthMouth1(shapes[i]));
        // features.push_back(heigthMouth2(shapes[i]));
        // features.push_back(tipLip_nose(shapes[i]));
        //*********************************************************
        features.push_back(openessMouth(shapes[i]));
        features.push_back(widthMouth(shapes[i]));
        features.push_back(widthEye(shapes[i]));
        features.push_back(nariz_sobrancelha1Mid(shapes[i]));
        features.push_back(nariz_sobrancelha2Mid(shapes[i]));
        features.push_back(alturaBoca1(shapes[i]));
        features.push_back(alturaBoca2(shapes[i]));
        features.push_back(aberturaOlho1(shapes[i]));
        features.push_back(aberturaOlho2(shapes[i]));
    }
    return features;
}

