// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.



    This face detector is made using the classic Histogram of Oriented
    Gradients (HOG) feature combined with a linear classifier, an image pyramid,
    and sliding window detection scheme.  The pose estimator was created by
    using dlib's implementation of the paper:
        One Millisecond Face Alignment with an Ensemble of Regression Trees by
        Vahid Kazemi and Josephine Sullivan, CVPR 2014
    and was trained on the iBUG 300-W face landmark dataset.

    Also, note that you can train your own models using dlib's machine learning
    tools.  See train_shape_predictor_ex.cpp to see an example.




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


#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <math.h>
#include <dlib/svm_threaded.h>
#include <dirent.h>
#include <fstream>
#include "myfeatures.h"
using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

//create a new blank file, and insert labels in column heading
void createSpread()
{
	ofstream file_descriptor;
    file_descriptor.open("../features.xls",ios::trunc);

    if (file_descriptor.is_open() && file_descriptor.good())
    {

        file_descriptor << "Openess Mouth\tWidth Mouth\tWidth Eye\t heigthEyebrow1\theigthEyebrow2\t heigthMouth1\t heigthMouth2\t tipLip_nose\tClass\n";

    }
    else
    {
    	cout << "Error in createSpread! Can't use file";
    	exit(1);
    }
    file_descriptor.close();

}

//insert a new feature line in file
void updateSpread(std::vector<double> feat, int feat_class)
{
	ofstream file_descriptor;

	file_descriptor.open("../features.xls",ios::app);

    if (file_descriptor.is_open() && file_descriptor.good())
    {
    	for (int i = 0; i < feat.size(); ++i)
	       file_descriptor<<feat[i]<<"\t";

        file_descriptor<<feat_class<<"\t";
        file_descriptor<<"\n";
    }
    else
    {
    	cout << "Error in updateSpread! Can't use file";
    	exit(1);
    }
    file_descriptor.close();
}



std::vector<full_object_detection>  image_processing(string path, shape_predictor sp)
{
	frontal_face_detector detector;
	array2d<rgb_pixel> img;
	std::vector<full_object_detection> shapes;
	try
    {
        detector = get_frontal_face_detector();

        cout << "processing image " << path << "...";

        load_image(img, path);
        // Make the image larger so we can detect small faces.
        pyramid_up(img);

        // Now tell the face detector to give us a list of bounding boxes
        // around all the faces in the image.
        std::vector<rectangle> dets = detector(img);
        // Now we will go ask the shape_predictor to tell us the pose of
        // each face we detected.

        for (unsigned long j = 0; j < dets.size(); ++j)
        {
            full_object_detection shape = sp(img, dets[j]);
            shapes.push_back(shape);
        }
    }

    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }


    return shapes;
}


int main(int argc, char** argv)
{
	//neutro = 0, feliz = 1, tristeza = 2, surpresa = 3, raiva = 4
	string emotions[5] = {"neutro", "feliz", "tristeza", "surpresa", "raiva"};
	int log[5] = {0,0,0,0,0};
	std::vector<full_object_detection> shapes;
	std::vector<double> feat;
	string fileName;
	shape_predictor sp;
	DIR *dir = NULL;
	struct dirent *pdir = NULL;



	try
    {
        // This example takes in a shape model file and then a list of images to
        // process.  We will take these filenames in as command line arguments.
        // Dlib comes with example images in the examples/faces folder so give
        // those as arguments to this program.
        if (argc == 0)
        {
            cout << "Call this program like this:" << endl;
            cout << "./face_landmark_detection_ex shape_predictor_68_face_landmarks.dat faces/*.jpg" << endl;
            cout << "\nYou can get the shape_predictor_68_face_landmarks.dat file from:\n";
            cout << "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
            return 0;
        }

        // We need a face detector.  We will use this to get bounding boxes for
        // each face in an image.

        // And we also need a shape_predictor.  This is the tool that will predict face
        // landmark positions given an image and face bounding box.  Here we are just
        // loading the model from the shape_predictor_68_face_landmarks.dat file you gave
        // as a command line argument.

        deserialize("../shape_predictor_68_face_landmarks.dat") >> sp;
        createSpread();


		for(int category = 0; category < 5; category++)
		{
			cout << "********Processing " + emotions[category] +  " pictures********" << endl;

			dir = opendir(("../database/" + emotions[category]).c_str());

			pdir = readdir(dir);

			while(pdir = readdir(dir))
			{
				if(pdir != NULL)
				{
					fileName = pdir->d_name;
					if(fileName.size() > 2)
					{
						++log[category];
						shapes = image_processing("../database/" + emotions[category] + "/" + fileName, sp);
						feat = featuresExtraction(shapes);
						updateSpread(feat, category);
						cout << " done." << endl;
					}
				}
			}

			closedir(dir);
		}

		cout << "Finish. " << log[0] + log[1] + log[2] + log[3] + log[4] << " Images processed." <<endl;
		cout<< "Images for category: " << endl;
		for(int i = 0; i < 5; i++)
		{
			cout << emotions[i] << ": " << log[i] << endl;
		}
        for (int i = 0; i < feat.size(); ++i){
            cout<<feat[i]<<" ";
        }
        cout<<endl;
		// cout << "Finish. " << log[0] + log[1] + log[2] + log[3] + log[4] << " Images processed." <<endl;
		// cout<< "Images for category: " << endl;
		// for(int i = 0; i < 5; i++)
		// {
		// 	cout << emotions[i] << ": " << log[i] << endl;
		// }

	}

    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
        exit(1);
    }


}

// ----------------------------------------------------------------------------------------

