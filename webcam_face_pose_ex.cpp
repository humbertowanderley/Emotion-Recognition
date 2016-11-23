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
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
// #include <dlib/gui_widgets.h>
#include <omp.h>
#include <string.h>
#include <Python.h>
#include "myfeatures.h"

using namespace dlib;
using namespace std;

#define BUFFER_SIZE 20
#define ALL_FRAME numDeslike+numLike+indiferent

double posx,posy;
string aux_text;
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
    draw_solid_circle (*imagem, shape.part(47),2, rgb_pixel(255,0,0));
    draw_solid_circle (*imagem, shape.part(40),2, rgb_pixel(255,0,0));
}


std::vector<int>  classify(std::vector<double> feat)
{
    PyObject *pName,*pFunc,*pModule,*pArgs,*pValue,*pList;
    std::vector<int> out;
    Py_Initialize(); //Inicializa interpretador de python
    pName = PyString_FromString("classifier");
    pModule = PyImport_Import(pName);
    Py_DECREF(pName);
    if (pModule != NULL){
        pFunc = PyObject_GetAttrString(pModule,"classifyEmotion");

        if (pFunc && PyCallable_Check(pFunc)){
            pList = PyList_New(feat.size());
            for (int i = 0; i < feat.size(); ++i){

                PyObject *num = PyFloat_FromDouble(feat[i]);
                if (!num) {
                    Py_DECREF(pList);
                    throw logic_error("Unable to allocate memory for Python list");
                }
                PyList_SET_ITEM(pList, i, num);
            }
            pArgs = PyTuple_New(1);
            PyTuple_SetItem(pArgs, 0, pList);

            pValue = PyObject_CallObject(pFunc,pArgs);  //chama função em python
            if (pValue != NULL){
                if (PyList_Check(pValue)){
                    for (int i = 0; i < NUM_CLASSES ; ++i){
                        PyObject* aux = PyList_GetItem(pValue,i);
                        out.push_back((int)PyLong_AsLong(aux));
                    }
                }
                else
                {
                    cout<<"Nao é uma lista"<<endl;
                }

                Py_DECREF(pValue);
            }
            else{
                cout<<"Erro no retorno da função\n";
                PyErr_Print();
                exit(0);
            }
            Py_DECREF(pList);
            Py_DECREF(pArgs);
            Py_DECREF(pValue);
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    }
    else
    {

        cout<<"Problema carregando modulo"<<endl;
        PyErr_Print();
        exit(0);
    }
    return out;
    // Py_Finalize();
}

void catchFrames(cv::Mat* fifo_frame,cv::VideoCapture cap)//,image_window* win)
{
    cv::Mat temp;
    for(int i = 0 ; i < BUFFER_SIZE;i++)
    {
        cap >> temp;
        cv_image<bgr_pixel> cimg(temp);
        // putText(temp,aux_text,cvPoint(120,10),6,2,cvScalar(0,255,255),2);
        // win->set_image(cimg);
        fifo_frame[i] = temp;
    }
}

double calcularPorcentagem(long int x)
{
    return x*100 / (double)BUFFER_SIZE;
}

int main()
{
    std::vector<double> feat;


    /**************Flags control*****************/
    long int numLike=0,numDeslike=0,indiferent=0;

    /* Buffer circular que para colocar
    frames processados pela webcam */
    cv::Mat fifo_frame[BUFFER_SIZE];
    /*
    ----Variáveis que auxiliam o buffer circular----
    pos é posição atual que será consumida pela thread
    ind indica em qual posição será colocado o novo frame*/
    int ind = 0;
    try
    {
        cv::VideoCapture cap(0);
        cap.set(CV_CAP_PROP_FRAME_WIDTH,640);
        cap.set(CV_CAP_PROP_FRAME_HEIGHT,480);

        if (!cap.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }

        // image_window win;

        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

        while(/*!win.is_closed()*/ 1)
        {
            catchFrames(fifo_frame,cap);//,&win);
            for (int pos = 0; pos < BUFFER_SIZE; ++pos){
                cv_image<bgr_pixel> cimg(fifo_frame[pos]);

                std::vector<rectangle> faces = detector(cimg);
                // Find the pose of each face.
                std::vector<full_object_detection> shapes;
                for (unsigned long i = 0; i < faces.size(); ++i)
                {
                    full_object_detection shape = pose_model(cimg, faces[i]);
                    posx=shape.part(9).x()-100;
                    posy=shape.part(9).y()+50;
                    chip_details chip = get_face_chip_details(shape,100);
                    shapes.push_back(map_det_to_chip(shape, chip));

                //    drawPoints(shape,&cimg);
                }
                feat = featuresExtraction(shapes);
                //print para debug
                /*for (int i = 0; i < feat.size();++i){
                    cout<<feat[i]<<" ";
                }*/
                //cout<<endl;
                std::vector<int> emotion = {0,0,0,0,0,0};
                cv::Mat gamb = toMat(cimg);
                for (int i = 0; i < feat.size(); i+=NUM_FEATURES){
                    std::vector<double> aux(feat.begin()+i,feat.begin()+i+(NUM_FEATURES));
                    emotion = classify(aux);


                    //Posição 5 é um único clasificador multiclasse abordagem 1 vs 1, criando 4 + 3 +2 + 1 SVMs para classificar 5 clases
                    switch(emotion[5])
                    {
                        case NEUTRAL: //cout << "Neutro" << endl;
                                // putText(gamb,"Neutro",cvPoint(posx,posy),6,2,cvScalar(0,255,255),2);
                                // aux_text = "Neutro";
                                indiferent++;
                            break;
                        case HAPPY: //cout << "Feliz" << endl;
                                // putText(gamb,"Feliz",cvPoint(posx,posy),6,2,cvScalar(0,255,255),2);
                                // aux_text = "Feliz";
                                numLike++;
                            break;
                        case SAD: //cout << "Triste" << endl;
                                 // putText(gamb,"Triste",cvPoint(posx,posy),6,2,cvScalar(0,255,255),2);
                                 // aux_text = "Triste";
                                 numDeslike++;
                            break;
                        case SURPRISE: //cout << "surpresa" << endl;
                                // putText(gamb,"Surpresa",cvPoint(posx,posy),6,2,cvScalar(0,255,255),2);
                                // aux_text = "Surpresa";
                                numLike++;
                            break;
                        case MAD: //cout << "Raiva" <<endl;
                                // putText(gamb,"Raiva",cvPoint(posx,posy),6,2,cvScalar(0,255,255),2);
                                // aux_text = "Raiva";
                                numDeslike++;
                                break;
                    }

                }
                // Display it all on the screen
                if (pos==BUFFER_SIZE-1){
                    aux_text="like="+to_string(calcularPorcentagem(numLike));
                    aux_text+="%%\ndelike="+to_string(calcularPorcentagem(numDeslike));
                    aux_text+="%%\nindiferent="+to_string(calcularPorcentagem(indiferent))+"%%";
                    cout<<aux_text<<endl;
                    // putText(gamb,aux_text,cvPoint(120,10),6,2,cvScalar(0,255,255),2);
                    numLike=numDeslike=indiferent=0;
                }
                // win.clear_overlay();
                // win.set_image(cimg);



                // win.add_overlay(render_face_detections(shapes));
            }

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
    return 0;
}