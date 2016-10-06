// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.


    This example is essentially just a version of the face_landmark_detection_ex.cpp
    example modified to use OpenCV's VideoCapture object to read from a camera instead
    of files.


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.
*/

#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <utility>

using namespace dlib;
using namespace std;

double length(point a,point b)
{
    return sqrt( (a.x()-b.x())*(a.x()-b.x()) + (a.y()-b.y())*(a.y()-b.y()) );
}
//Features
double openessMouth(full_object_detection shape)    //33 - happy
{
    return length(shape.part(51),shape.part(57));
}

double widthMouth(full_object_detection shape)      //
{
    return length(shape.part(48),shape.part(54));
}

double widthEye(full_object_detection shape)      //
{
    return length(shape.part(36),shape.part(39))/2.0 + length(shape.part(42),
        shape.part(45))/2.0;
}

double heigthEyebrow1(full_object_detection shape)      //

    return length(shape.part(30),shape.part(17))/2.0 + length(shape.part(30),
        shape.part(26))/2.0;
}

double heigthEyebrow2(full_object_detection shape)      //
{
    return length(shape.part(30),shape.part(21))/2.0 + length(shape.part(30),
        shape.part(22))/2.0;
}

double tipLip_nose(full_object_detection shape) //dist. do nariz à ponta da boca
{
    return length(shape.part(30),shape.part(21))/2.0 + length(shape.part(30),shape.part(22))/2.0;
}

//Other things
std::vector<float> featuresExtraction(full_object_detection shape)
{
    std::vector<float> features;
    for (int i = 0; i < shape.num_parts(); ++i)
    {
        // l[i] = make_pair((float) shape.part(i).x(),(float) shape.part(i).y());
        features.push_back(openessMouth(shape[i]));
        features.push_back(widthMouth(shape[i]));
        features.push_back(widthEye(shape[i]));
        features.push_back(heigthEyebrow1(shape[i]));
        features.push_back(heigthEyebrow2(shape[i]));
        features.push_back(tipLip_nose(shape[i]));
        features.push_back(-1.0);
    }

    return features;
}

//função usada como debug para descobrir onde estão os pontos
void drawPoints(full_object_detection shape, cv_image<bgr_pixel> *imagem)
{
    for(int i = 0; i < shape.num_parts(); i++)
    {
        //desenha landmarks
        draw_solid_circle (*imagem, shape.part(i),2, rgb_pixel(0,255,0));
    }

    //draw_solid_circle (*imagem, shape.part(48),2, rgb_pixel(255,0,0));
    //draw_solid_circle (*imagem, shape.part(54),2, rgb_pixel(255,0,0));
    draw_solid_circle (*imagem, shape.part(36),2, rgb_pixel(255,0,0));
    draw_solid_circle (*imagem, shape.part(39),2, rgb_pixel(255,0,0));
}


int main()
{

    cv::Mat temp;

    try
    {
        cv::VideoCapture cap(0);
        if (!cap.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }

        image_window win;

        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

        // Grab and process frames until the main window is closed by the user.
        while(!win.is_closed())
        {
            // Grab a frame
            cap >> temp;
            // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
            // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
            // long as temp is valid.  Also don't do anything to temp that would cause it
            // to reallocate the memory which stores the image as that will make cimg
            // contain dangling pointers.  This basically means you shouldn't modify temp
            // while using cimg.
            cv_image<bgr_pixel> cimg(temp);

            // Detect faces
            std::vector<rectangle> faces = detector(cimg);
            // Find the pose of each face.]

            std::vector<full_object_detection> shapes;
            for (unsigned long i = 0; i < faces.size(); ++i)
            {
                shapes.push_back(pose_model(cimg, faces[i]));
                drawPoints(shapes[i],&cimg);
                //cout<<"Abertura da boca da face "<<i<<":"<<openessMouth(shapes[i]);
                //cout<<"Largura da boca:"<<widthMouth(shapes[i]);
                //featuresExtraction(shapes[i]);
            }



            // Display it all on the screen
            win.clear_overlay();
            win.set_image(cimg);
            //win.add_overlay(render_face_detections(shapes));
           /* cout<<"Vetor de landmarks: ";

            for (int i = 0; i < 68 ; ++i){
                cout<<"("<<vx[i]<<","<<vy[i]<<"), ";
                // norm[i] = (landmarks[i]-min(landmarks))/(max(landmarks)-min(landmarks)) ;
            }
*/
            cout<<" "<<endl;
        }
    }
    catch(serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
    }
}

