CXX ?= g++
CFLAGS = -Wall -Wconversion -O3 -fPIC

all: createLabelledDataset calculateHoGFeatures createTrainingData predictImage detectClassImage createTrainingDataPCA detectClassImagePCA detectClassImageFast detectClassImageFastPCA

createLabelledDataset: createLabelledDataset.cpp
	g++ -o createLabelledDataset createLabelledDataset.cpp `pkg-config --libs opencv`

calculateHoGFeatures: calculateHoGFeatures.cpp
	g++ -o calculateHoGFeatures calculateHoGFeatures.cpp `pkg-config --libs opencv`

createTrainingData: createTrainingData.cpp
	g++ -o createTrainingData createTrainingData.cpp `pkg-config --libs opencv`

createTrainingDataPCA: PCA.o createTrainingDataPCA.cpp
	g++ -o createTrainingDataPCA createTrainingDataPCA.cpp PCA.o `pkg-config --libs opencv`

predictImage: svmPredictVector.o predictImage.cpp
	g++ -o predictImage predictImage.cpp svmPredictVector.o svm.o `pkg-config --libs opencv`

detectClassImage: svmPredictVector.o detectClassImage.cpp
	g++ -o detectClassImage detectClassImage.cpp svmPredictVector.o svm.o `pkg-config --libs opencv`

detectClassImagePCA: PCA.o svmPredictVector.o detectClassImagePCA.cpp
	g++ -o detectClassImagePCA detectClassImagePCA.cpp svmPredictVector.o svm.o PCA.o `pkg-config --libs opencv`

detectClassImageFast: HOG.o svmPredictVector.o detectClassImageFast.cpp
	g++ -o detectClassImageFast detectClassImageFast.cpp svmPredictVector.o svm.o HOG.o `pkg-config --libs opencv`

detectClassImageFastPCA: HOG.o PCA.o svmPredictVector.o detectClassImageFastPCA.cpp
	g++ -o detectClassImageFastPCA detectClassImageFastPCA.cpp svmPredictVector.o svm.o PCA.o HOG.o `pkg-config --libs opencv`

svmPredictVector.o: svm.o svmPredictVector.h svmPredictVector.cpp
	g++ -c svmPredictVector.cpp

HOG.o: HOG.cpp HOG.h
	g++ -c HOG.cpp

PCA.o: PCA.cpp PCA.h
	g++ -c PCA.cpp

svm.o: svm.h svm.cpp
	$(CXX) $(CFLAGS) -c svm.cpp

clean:
	rm -rf createLabelledDataset
	rm -rf calculateHoGFeatures
	rm -rf createTrainingData
	rm -rf createTrainingDataPCA
	rm -rf detectClassImagePCA
	rm -rf predictImage
	rm -rf detectClassImage
	rm -rf detectClassImageFast
	rm -rf detectClassImageFastPCA
	rm -rf svmPredictVector.o
	rm -rf PCA.o
	rm -rf HOG.o
	rm -rf svm.o