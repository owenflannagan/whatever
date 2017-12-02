#include <fstream>
#include <vector>
#include <iostream>
#include <random>
#include "Eigen/Dense"
#include "nnimage.h"
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace Eigen;
using namespace cv;

srand(1);

/* Helper library
 */
class mnist_loader {
public:
    static vector<nnimage> load(std::string imgs_name, std::string labels_name, int num, bool train) {
        cout << "Loading data..." << endl;
        ifstream imgs(imgs_name, ios::binary);
        ifstream labels(labels_name, ios::binary);

        vector<nnimage> images;

        double rNum;
        int checksum = 0;
        int image_count = 0;
        int image_width = 0;
        int image_height = 0;

        int label_checksum = 0;
        int label_count = 0;

        imgs.read((char*)&checksum, sizeof(checksum));
        imgs.read((char*)&image_count, sizeof(image_count));
        imgs.read((char*)&image_width, sizeof(image_width));
        imgs.read((char*)&image_height, sizeof(image_height));


        labels.read((char*)&label_checksum, sizeof(label_checksum));
        labels.read((char*)&label_count, sizeof(label_count));

        for (int i = 0; i < num; i++) {
            if (i % 5000 == 0) {
                cout << "Loading: " << i << " / " << num << " completed." << endl;
            }
            MatrixXd data(784, 1);

            unsigned char label = 0;
            labels.read((char*)&label, sizeof(label));

            //MatrixXd dataUp(784, 1);
            //MatrixXd dataDown(784, 1);
            //MatrixXd dataLeft(784, 1);
            //MatrixXd dataRight(784, 1);
            MatrixXd dataRotate;
            MatrixXd dataShearRight(28, 28);
            MatrixXd dataShearLeft(28, 28);


            /*for (int i = 0; i < 56; i++) {
                dataDown(i, 0) = 0.0;
                if (i < 28) {
                    dataRight((i * 28), 0) = 0.0;
                    dataRight(((i * 28) + 1), 0) = 0.0;
                    dataLeft(((i * 28) + 26), 0) = 0.0;
                    dataLeft(((i * 28) + 27), 0) = 0.0;
                }
            }*/
            //cout << "Test 1" << endl;

            int row = 0;
            int deduct = 0;
            int col = 0;
            for (int j = 0; j < 784; j++) {
                unsigned char c = 0;
                imgs.read((char*)&c, sizeof(c));
                data(j, 0) = double(c);
                data(j, 0) = data(j, 0) / 1000.0;
                /*if(j < 728) {
                    dataDown((j+56), 0) = double(c);
                    dataDown((j+56), 0) /= 1000.0;
                }
                if(j >= 56) {
                    dataUp((j-56), 0) = double(c);
                    dataUp((j-56), 0) /= 1000.0;
                }
                if((j % 28) < 26) {
                    dataRight((j+2), 0) = double(c);
                    dataRight((j+2), 0) /= 1000.0;
                }
                if((j % 28) > 1) {
                    dataLeft((j-2), 0) = double(c);
                    dataLeft((j-2), 0) /= 1000.0;
                }*/
                col = j - deduct;
                dataTurnClockwise(row, col) = double(c);
                dataTurnClockwise(row, col) /= 1000.0;
                if(j != 0 && ((j+1) % 28 == 0)) {
                    row++;
                    cout << "Maybe" << endl;
                    deduct += 28;
                }
            }

            /*for(int i = 728; i < 784; i++) {
                dataUp(i, 0) = 0.0;
            }*/

            dataTurnCounterClockwise = dataTurnClockwise;

            //rotation
            dataRotate = rotate(data);

            //perspective changes
            rNum = (double)rand() / RAND_MAX;
            dataShearRight = shearHorizontal(dataTurnClockwise, rNum);
            rNum = (double)rand() / RAND_MAX;
            dataShearLeft = shearVertical(dataTurnClockwise, rNum);
            dataZoomHorizontal = loadAffine(dataTurnClockwise);
            dataZoomVertical = loadAffine(dataTurnClockwise);

            nnimage nnimg(data, (int)label);
            images.push_back(nnimg);
            /*nnimage nnimgUp(dataUp, (int)label);
            images.push_back(nnimgUp);
            nnimage nnimgDown(dataDown, (int)label);
            images.push_back(nnimgDown);
            nnimage nnimgLeft(dataLeft, (int)label);
            images.push_back(nnimgLeft);
            nnimage nnimgRight(dataRight, (int)label);
            images.push_back(nnimgRight);*/
            if (!train) {
                nnimage nnimgClock(clockFinal, (int)label);
                images.push_back(clockFinal);
                nnimage nnimgCounter(counterFinal, (int)label);
                images.push_back(counterFinal);
            }
        }

        imgs.close();
        cout << "Data loaded." << endl;
        return images;
    }

    static Mat loadAffine(Mat sourceMat, int x1, int y1, int x2, int y2,
                                    int x3, int y3, int x4, int y4) {
        Mat source;
        Mat dest;
        Mat lambda;
        Point2f input[4];
        Point2f output[4];
        input[0] = Point2f(0, 0);
        input[1] = Point2f(source.cols - 1, 0);
        input[2] = Point2f(0, source.rows - 1);
        input[3] = Point2f(source.cols - 1, source.rows - 1);
        output[0] = Point2f(x1, y1);
        output[1] = Point2f(x2, y2);
        output[2] = Point2f(x3, y3);
        output[3] = Point2f(x4, y4);
        lambda = getPerspectiveTransform(input, output);
        warpPerspective(source, dest, lambda, source.size());
        return dest;
    }

    static MatrixXd shearHorizontal(MatrixXd source, double proportion) {
        Mat sourceMat;
        eigen2cv(source, sourceMat);
        sourceMat = sourceMat.reshape(28, 28);
        int pixels = (int)((proportion * sourceMat.cols()) / 2);
        Mat destMat;
        width = sourceMat.cols();
        length = sourceMat.rows();
        destMat = loadAffine(sourceMat, pixels, 0, width+pixels, 0,
                            width-pixels, length, -pixels, length);
        destMat = destMat.reshape(1, 784);
        MatrixXd dest;
        cv2eigen(destMat, dest);
        return dest;
    }

    static MatrixXd shearVertical(MatrixXd source, double proportion) {
        Mat sourceMat;
        eigen2cv(source, sourceMat);
        sourceMat = sourceMat.reshape(28, 28);
        int pixels = (int)((proportion * sourceMat.rows()) / 2);
        Mat destMat;
        width = sourceMat.cols();
        length = sourceMat.rows();
        destMat = loadAffine(sourceMat, 0, pixels, width, -pixels,
                            width, length-pixels, 0, length+pixels);
        destMat = destMat.reshape(1, 784);
        MatrixXd dest;
        cv2eigen(destMat, dest);
        return dest;
    }

    static double randRange(double low, double high) {
        double rNum = (double)rand() / RAND_MAX;
        double range = high - low;
        double ret = (rNum * range) + low;
        return ret;
    }

    static MatrixXd rotate(MatrixXd source, double rNum) {
        Mat sourceMat;
        eigen2cv(source, sourceMat);
        sourceMat = sourceMat.reshape(28, 28);
        Point2f center(sourceMat.cols/2, sourceMat.rows/2);
        Mat rotate = getRotationMatrix2D(center, rNum, 1.0);
        Mat destMat;
        warpAffine(sourceMat, destMat, rotate, sourceMat.size());
        destMat = destMat.reshape(1, 784);
        MatrixXd dest;
        cv2eigen(destMat, dest);
        return dest;
    }

    static MatrixXd zoomVertical(MatrixXd source, double proportion) {
        Mat sourceMat;
        eigen2cv(source, sourceMat);
        sourceMat = sourceMat.reshape(28, 28);
        int pixels = (int)(proportion * sourceMat.cols());
        Mat destMat;
        width = sourceMat.cols();
        length = sourceMat.rows();
        destMat = loadAffine(sourceMat, 0, -pixels, width, -pixels,
                            width, length + pixels, 0, length + pixels);
        destMat = destMat.reshape(1, 784);
        MatrixXd dest;
        cv2eigen(destMat, dest);
        return dest;
    }

    static MatrixXd zoomHorizontal(MatrixXd source, double proportion) {
        Mat sourceMat;
        eigen2cv(source, sourceMat);
        sourceMat = sourceMat.reshape(28, 28);
        int pixels = (int)(proportion * sourceMat.cols());
        Mat destMat;
        width = sourceMat.cols();
        length = sourceMat.rows();
        destMat = loadAffine(sourceMat, -pixels, 0, width + pixels, 0,
                            width + pixels, length, -pixels, length);
        destMat = destMat.reshape(1, 784);
        MatrixXd dest;
        cv2eigen(destMat, dest);
        return dest;
    }
};
