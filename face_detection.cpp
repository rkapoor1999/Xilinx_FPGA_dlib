#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <time.h>
using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

// The next bit of code defines a ResNet network.  It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and made the network somewhat smaller.  Go read the introductory
// dlib DNN examples to learn what all this stuff means.
//
// Also, the dnn_metric_learning_on_images_ex.cpp example shows how to train this network.
// The dlib_face_recognition_resnet_model_v1 model used by this example was trained using
// essentially the code shown in dnn_metric_learning_on_images_ex.cpp except the
// mini-batches were made larger (35x15 instead of 5x5), the iterations without progress
// was set to 10000, and the training dataset consisted of about 3 million images instead of
// 55.  Also, the input layer was locked to images of size 150.
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

//----------------------------------------------------------------------------------------

std::vector<matrix<rgb_pixel>> jitter_image(
    const matrix<rgb_pixel>& img
);

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv) try
{
    if (argc != 5)
    {
        cout << "Run this example by invoking it like this: " << endl;
        cout << "   ./face_detection bald_guys.jpg another_image.jpg string1 string2" << endl;
        cout << endl;
        cout << "You will also need to get the face landmarking model file as well as " << endl;
        cout << "the face recognition model file.  Download and then decompress these files from: " << endl;
        cout << "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2" << endl;
        cout << "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2" << endl;
        cout << endl;
        return 1;
    }
    cout<<argc<<endl;
       clock_t start_t1, end_t1, start_t2, end_t2;
       double 	total_t1, total_t2;
       //int iterations=0;
    // The first thing we are going to do is load all our models.  First, since we need to
    // find faces in the image we will need a face detector:
    frontal_face_detector detector = get_frontal_face_detector();
    // We will also use a face landmarking model to align faces to a standard pose:  (see face_landmark_detection_ex.cpp for an introduction)
    shape_predictor sp;
    deserialize("shape_predictor_5_face_landmarks.dat") >> sp;
    // And finally we load the DNN responsible for face recognition.
    anet_type net;
    deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

    matrix<rgb_pixel> img;
    matrix<rgb_pixel> img_2;
    load_image(img, argv[1]);
    load_image(img_2, argv[2]);
    // Display the raw image on the screen
    //image_window win(img); 

    // Run the face detector on the image of our action heroes, and for each face extract a
    // copy that has been normalized to 150x150 pixels in size and appropriately rotated
    // and centered.
    //std::vector<matrix<rgb_pixel>> faces;
    //Removed the for and implemented the idea without the break. Detector is a container containing strings each of which has image data for each face detected. saved in face the data for the first
    //Removed the use of vector, using matrix instead. Made neccesary changes ahead also to incorporate it.
    	matrix<rgb_pixel> faces, faces_2;
    	
    	start_t1=clock();
    	auto face = detector(img)[0];
        auto shape = sp(img, face);
        extract_image_chip(img, get_face_chip_details(shape,150,0.25), faces);
		end_t1=clock();
		total_t1=1000*(double)(end_t1-start_t1)/CLOCKS_PER_SEC;
		
		start_t2 = clock();
		auto face_2 = detector(img_2)[0];
		auto shape_2 = sp(img_2, face_2);
		extract_image_chip(img_2, get_face_chip_details(shape_2,150,0.25), faces_2);
		end_t2 = clock();
		total_t2=1000*(double)(end_t2-start_t2)/CLOCKS_PER_SEC;
        //faces.push_back(move(face_chip));
        // Also put some boxes on the faces so we can see that the detector is finding
        // them.
        //win.add_overlay(face);
   
    printf("Time taken to detect and reshape the first face in image 1: %fms\n",total_t1);
    printf("Time taken to detect and reshape the first face in image 2: %fms\n",total_t2);
    
    if (faces.size() == 0)
    {
        cout << "No faces found in image 1" << endl;
        return 1;
    }
    if (faces.size() == 0)
    {
        cout << "No faces found in image 2" << endl;
        return 1;
    }

    // This call asks the DNN to convert each face image in faces into a 128D vector.
    // In this 128D vector space, images from the same person will be close to each other
    // but vectors from different people will be far apart.  So we can use these vectors to
    // identify if a pair of images are from the same person or from different people.  
    start_t2=clock();
	matrix<float,0,1> face_descriptors = net(faces);
	matrix<float,0,1> face_descriptors_2 = net(faces_2);
	end_t2=clock();
	total_t2=1000*(double)(end_t2-start_t2)/CLOCKS_PER_SEC;
    printf("128D convert time %fms\n",total_t2);
    cout << "Face found in first image?(1 for yes, 0 for no)" << face_descriptors.size()/128<<"\n";
    cout << "Face found in second image?(1 for yes, 0 for no)" << face_descriptors_2.size()/128<<"\n";
    //lines 129 to 137 creates a new data file in which it saves the detected face data which is stored in face_descriptors matrix
    string filename = argv[3];
    ofstream output(filename + "." + "dat");
	output << face_descriptors(1,0) << "\n";
    ofstream zOut(filename + "." + "dat", ios::out | ios::binary);
    for (size_t x = 0; x < face_descriptors; x++){   
		zOut.write(reinterpret_cast<char*> (&face_descriptors(x,0)), sizeof(float));
   	}
	zOut.close();
	cout<<"\nFace Detection data saved by the name \""<<filename<<".dat\" in the directory the code is running from";
	
	string filename_2 = argv[4];
    ofstream output_2(filename_2 + "." + "dat");
	output_2 << face_descriptors_2(1,0) << "\n";
    ofstream zOut_2(filename_2 + "." + "dat", ios::out | ios::binary);
    for (size_t x = 0; x < face_descriptors_2; x++){   
		zOut_2.write(reinterpret_cast<char*> (&face_descriptors_2(x,0)), sizeof(float));
   	}
	zOut_2.close();
	cout<<"\nFace Detection data saved by the name \""<<filename_2<<".dat\" in the directory the code is running from";
	
	
	cout << "\nhit enter to terminate" << endl;
    cin.get();
}

catch (exception& e)
{
    cout << e.what() << endl;
}

// ----------------------------------------------------------------------------------------

std::vector<matrix<rgb_pixel>> jitter_image(
    const matrix<rgb_pixel>& img
)
{
    // All this function does is make 100 copies of img, all slightly jittered by being
    // zoomed, rotated, and translated a little bit differently. They are also randomly
    // mirrored left to right.
    thread_local dlib::rand rnd;

    std::vector<matrix<rgb_pixel>> crops; 
    for (int i = 0; i < 100; ++i)
        crops.push_back(jitter_image(img,rnd));

    return crops;
}

// ----------------------------------------------------------------------------------------

