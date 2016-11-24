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
#include <dlib/gui_widgets.h>
#include <omp.h>
#include <Python.h>
#include "myfeatures.h"

using namespace dlib;
using namespace std;

#define BUFFER_SIZE 5
#define POS_X 10
#define POS_Y 50
#define NUM_PESSOAS numDeslike+numLike+indiferent

enum {
    NEUTRAL=0,
    HAPPY,
    SAD,
    SURPRISE,
    MAD
 };

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


std::vector<int> classify(std::vector<double> feat)
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


double calcularPorcentagem(int x,int y)
{
    return x*100 / (double)y;
}

int maiorEmocao(int* buffer,int iter)
{
    int v[] = {0,0,0,0,0};
    for (int i = 0; i < iter; ++i){
        //Posição 5 é um único classificador multiclasse abordagem 1 vs 1, criando 4 + 3 +2 + 1 SVMs para classificar 5 clases
        switch(buffer[i])
        {
            case NEUTRAL:
                    v[NEUTRAL]++;
                break;
            case HAPPY:
                    v[HAPPY]++;
                break;
            case SAD:
                    v[SAD]++;
                break;
            case SURPRISE:
                    v[SURPRISE]++;
                break;
            case MAD:
                    v[MAD]++;
                break;
        }
    }
    cout<<"VETOR\n";
    int max = *max_element(v,v+5);
    for (int i = 0; i < 5; ++i){
        if (v[i]==max)
            return i;
    }
}

int main()
{
    cv::Mat temp;
    std::vector<double> feat;
    /* Buffer circular que para colocar
    frames processados pela webcam */
    cv::Mat aux_mat;
    string aux_text="",like_text="",deslike_text="",ind_text="";

     /**************Flags control*****************/
    int numLike=0,numDeslike=0,indiferent=0;
    // variável q espera o cara que ja foi classificado sair
    bool wait = 0;
    int buffer[BUFFER_SIZE];
    int iter = 0;
    // cv::Mat fifo_frame[BUFFER_SIZE];
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

        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
        image_window win;

        // uma nova thread é criada junto a master
        #pragma omp parallel num_threads(2)
        {
            // indica qual é a thread. 0->master/producer,1->consumer
            int me = omp_get_thread_num();
            while(!win.is_closed())
            {
                #pragma omp barrier
                if (!me){       // if it is master and producer
                    while(ind < BUFFER_SIZE)
                    {
                        cout<<"capture "<<ind<<"    "<<iter<<endl;
                        cap >> temp;
                        aux_mat = temp;
                        cv_image<bgr_pixel> cimg(temp);
                        putText(temp,like_text,cvPoint(POS_X,POS_Y),6,1,cvScalar(0,255,255),2);
                        putText(temp,deslike_text,cvPoint(POS_X,POS_Y+40),6,1,cvScalar(0,255,255),2);
                        putText(temp,ind_text,cvPoint(POS_X,POS_Y+80),6,1,cvScalar(0,255,255),2);
                        win.set_image(cimg);
                        ind++;
                    }

                }
                else{           // if it is the consumer
                    if (ind == BUFFER_SIZE){
                        cv_image<bgr_pixel> cimg(aux_mat);

                        std::vector<rectangle> faces = detector(cimg);
                        // Find the pose of each face.
                        std::vector<full_object_detection> shapes;
                        for (unsigned long i = 0; i < faces.size(); ++i)
                        {
                            full_object_detection shape = pose_model(cimg, faces[i]);
                            chip_details chip = get_face_chip_details(shape,100);
                            shapes.push_back(map_det_to_chip(shape, chip));
                        }
                        feat = featuresExtraction(shapes);
                        //print para debug
                        /*for (int i = 0; i < feat.size();++i){
                            cout<<feat[i]<<" ";
                        }
                        cout<<endl;*/
                        if (wait && feat.size()==0)
                        {
                            cout<<"sai do wait\n";
                            wait = 0;
                        }
                        std::vector<int> emotion = {0,0,0,0,0,0};
                        for (int i = 0; i < feat.size(); i+=NUM_FEATURES){
                            std::vector<double> aux(feat.begin()+i,feat.begin()+i+(NUM_FEATURES));
                            emotion = classify(aux);
                            if(!wait)
                                buffer[(iter++)%BUFFER_SIZE] = emotion[5];
                        }

                        if(iter >= BUFFER_SIZE){

                            switch(maiorEmocao(buffer,iter))
                            {
                                case NEUTRAL:
                                        cout<<"maior emocao foi neutro"<<endl;
                                        indiferent++;
                                    break;
                                case HAPPY:
                                        cout<<"maior emocao foi feliz"<<endl;
                                        numLike++;
                                    break;
                                case SAD:
                                        cout<<"maior emocao foi triste"<<endl;
                                        numDeslike++;
                                    break;
                                case SURPRISE:
                                        cout<<"maior emocao foi surpresa"<<endl;
                                        numLike++;
                                    break;
                                case MAD:
                                        cout<<"maior emocao foi raiva"<<endl;
                                        numDeslike++;
                                    break;
                            }

                            cv::Mat gamb = toMat(cimg);
                            ;
                            like_text="like="+to_string(calcularPorcentagem(numLike,NUM_PESSOAS))+"%%";
                            putText(gamb,like_text,cvPoint(POS_X,POS_Y),6,1,cvScalar(0,255,255),2);
                            deslike_text ="deslike="+to_string(calcularPorcentagem(numDeslike,NUM_PESSOAS))+"%%";
                            putText(gamb,deslike_text,cvPoint(POS_X,POS_Y+40),6,1,cvScalar(0,255,255),2);
                            ind_text="indiferent="+to_string(calcularPorcentagem(indiferent,NUM_PESSOAS))+"%%";
                            putText(gamb,ind_text,cvPoint(POS_X,POS_Y+80),6,1,cvScalar(0,255,255),2);
                            cout<<"qtd pessoas="<<NUM_PESSOAS<<endl;
                            cout<<like_text<<" \t"<<deslike_text<<" \t"<<ind_text<<endl;
                            iter=0;
                            wait=1;
                            cout<<"entra no wait\n";
                            // Display it all on the screen
                            win.clear_overlay();
                            win.set_image(cimg);
                            // win.add_overlay(render_face_detections(shapes));

                        }
                        ind = 0;
                    }
                }
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