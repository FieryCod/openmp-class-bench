//#include "Bench.h"
#include "rapidcsv.h"
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <array>
#include<stdio.h>
#include<stdlib.h>

using namespace std;

static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

struct nearestPointRow {
    double distance;
    double value;
};

void spit_csv(std::string filename, std::vector<std::vector<double>> ds, std::vector<std::string> cnames) {
    std::ofstream out;
    out.open(filename);

    for (auto name : cnames) {
        out << name << ",";
    }

    out << "\n";

    for (int i = 0; i < ds[0].size(); ++i) {
        for (int j = 0; j < cnames.size(); ++j) {
            out << ds[j][i] << ((j == cnames.size() - 1) ? "\n" : ",");
        }
    }

    out.close();
}

namespace knn {

    __device__ __host__ double euqlidesDistance(double *testPoint, double *learnPoint, unsigned n) {
        double sum = 0;
        for (int i = 0; i < n; ++i) {
            sum += pow(testPoint[i] - *(learnPoint + i), 2);
        }
        return sqrt(sum);
    }
    //TODO add other distance calculations

    __device__ void
    checkData(nearestPointRow *nearestPoint, unsigned startIndx, const double distance, const double value,
              const int neighbours) {
        for (int i = 0; i < neighbours; ++i) {
            if (nearestPoint[startIndx + i].distance > distance) {
                for (int j = (neighbours - 1); j > i; --j) {
                    nearestPoint[startIndx + j] = nearestPoint[startIndx + j - 1];
                }
                nearestPoint[startIndx + i].distance = distance;
                nearestPoint[startIndx + i].value = value;
            }
        }
    }

    double getGreatestValue(nearestPointRow *nearestPoint, unsigned neighbours) {
        map<double, int> countMap;

        for (int i = 0; i < neighbours; ++i) {
            countMap[nearestPoint[i].value]++;
        }

        double best;
        int bestC = 0;
        for (auto x : countMap) {
            if (bestC < x.second) {
                bestC = x.second;
                best = x.first;
            }
        }

        return best;
    }

    __global__ void
    process(double *learnData, unsigned learnDataSize, unsigned learnDataRows, nearestPointRow *nearestPoint,
            unsigned nearestPointSize, unsigned nearestPointRows, double *testData, unsigned perThread,
            unsigned short neighbours = 1) {
        unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned stride = blockDim.x * gridDim.x;
        unsigned reali, npi = threadIdx.x * gridDim.x + blockIdx.x;
        for (unsigned i = index; i < learnDataSize; i += stride) {
            reali = i * learnDataSize / learnDataRows;
            knn::checkData(nearestPoint, npi, knn::euqlidesDistance(testData, &learnData[reali], neighbours),
                           learnData[reali + 3],
                           neighbours);
        }

    }

    __host__ __device__ void sortNearestPoint(nearestPointRow *nearestPointsInput,
                                              unsigned indexStart, unsigned indexEnd) {
        nearestPointRow temp;
        for (unsigned i = indexStart + 1; i < indexEnd; ++i) {
            if (nearestPointsInput[i - 1].distance > nearestPointsInput[i].distance) {
                temp = nearestPointsInput[i - 1];
                nearestPointsInput[i - 1] = nearestPointsInput[i];
                nearestPointsInput[i] = temp;

                for (int j = i - 1; j > indexStart; --j) {
                    if (nearestPointsInput[j - 1].distance > nearestPointsInput[j].distance) {
                        temp = nearestPointsInput[j - 1];
                        nearestPointsInput[j - 1] = nearestPointsInput[j];
                        nearestPointsInput[j] = temp;
                    } else
                        break;
                }
            }
        }
    }

    __global__ void sortNearestPointRun(nearestPointRow *nearestPointsInput, unsigned perThread, unsigned size) {
        unsigned stride = blockDim.x * gridDim.x;
        int start = threadIdx.x * perThread * 2 + blockIdx.x * perThread;
        sortNearestPoint(nearestPointsInput, start, min(size, start + perThread));
    }

    int compare(const nearestPointRow* a, const nearestPointRow* b){
        if( a->distance > b->distance )
            return - 1;

        if( a->distance < b->distance)
            return 1;

        return 0;
    }

    double
    testRecord(double *dev_learnData, nearestPointRow *dev_nearestPointReturn, nearestPointRow *nearestPointInitial,
               array<double, 4> testData, unsigned dSize, unsigned dRows, unsigned npSize, unsigned npRows,
               int numBlocks, int blockSize, unsigned perThread, unsigned short neighbours = 1) {
        double *dev_test;
        HandleError(cudaMalloc(&dev_test, 4 * sizeof(double)));
        HandleError(cudaMemcpy(dev_test, testData.data(), 4 * sizeof(double), cudaMemcpyHostToDevice));

        HandleError(cudaMemcpy(dev_nearestPointReturn, nearestPointInitial, npSize * sizeof(nearestPointRow),
                                   cudaMemcpyHostToDevice));

        knn::process<<<numBlocks, blockSize>>>(dev_learnData, dSize, dRows, dev_nearestPointReturn, npSize, npRows,
                dev_test, perThread, neighbours);
        cudaDeviceSynchronize();

        nearestPointRow *nearestPointResult, *nearestPointResult2;
        unsigned tasksPerThread = neighbours * 200;
        unsigned numBlocks1 = ((npSize + tasksPerThread - 1) / tasksPerThread + blockSize - 1) / blockSize;;

        sortNearestPointRun<<<numBlocks, numBlocks1>>>(dev_nearestPointReturn, tasksPerThread, npSize);
        nearestPointResult2 = new nearestPointRow[numBlocks * blockSize * neighbours];
        nearestPointResult = new nearestPointRow[npSize];

        cudaDeviceSynchronize();

        HandleError(
                cudaMemcpy(nearestPointResult, dev_nearestPointReturn, npSize * sizeof(nearestPointRow),
                           cudaMemcpyDeviceToHost));

        for(unsigned irow = 0, npr2i = 0; irow < npSize; irow += perThread * neighbours){
            for (unsigned in = 0; in < neighbours; ++in) {
                nearestPointResult2[npr2i] = nearestPointResult[irow + in];
                ++npr2i;
            }
        }

        sortNearestPoint(nearestPointResult2, 0, numBlocks * blockSize * neighbours);

        double res = getGreatestValue(nearestPointResult, neighbours);
        delete[] nearestPointResult;
        delete[] nearestPointResult2;
        cudaFree(dev_test);
        return res;
    }

};

int main(int argc, char *argv[]) {
    if (argc != 5) {
        cout << "Bad parameters!" << endl;
        exit(1);
    }

    int THREADS = std::stoi(argv[2]);
    int BLOCKS = std::stoi(argv[3]);
    int TB_SWITCH = std::stoi(argv[4]);

    string file_path = string(argv[1]);
//TODO add parameter to chose neighbours
    int neighbours = 5;

    std::cout << "BLOCKS: " << BLOCKS
              << "\nTHREADS: " << THREADS
              << "\nCOUNT: " << THREADS * BLOCKS
              << "\n\n---------------------------------------\n";

    ifstream f(file_path);
    if (!f.good()) {
        cout << "File not found" << endl;
        exit(1);
    }
    f.close();

    rapidcsv::Document doc(file_path);

    vector<double> R = doc.GetColumn<double>("R");
    vector<double> G = doc.GetColumn<double>("G");
    vector<double> B = doc.GetColumn<double>("B");
    vector<double> SKIN = doc.GetColumn<double>("SKIN");

    const unsigned learnSize = R.size() * 0.8 * 4, learnRows = R.size() * 0.8;
    double *learn = new double[learnSize];
    double *dev_learn;
    vector<array<double, 4>> test;
    for (int i = 0; i < R.size(); ++i) {
        if (i < R.size() * 0.8) {
            learn[i * 4 + 0] = R[i];
            learn[i * 4 + 1] = G[i];
            learn[i * 4 + 2] = B[i];
            learn[i * 4 + 3] = SKIN[i];
        } else {
            test.push_back({R[i], G[i], B[i], SKIN[i]});
        }
    }
    /**
   * Create global bencher
   */
    std::unique_ptr<Bench> bencher = std::make_unique<Bench>(Bench());


    int tasksPerThread = neighbours * 10;
    int blockSize = 1024;//512
    unsigned numBlocks = ((learnRows + tasksPerThread - 1) / tasksPerThread + blockSize - 1) / blockSize;

    const unsigned npSize = blockSize * numBlocks * neighbours, npRows = blockSize * numBlocks;
    nearestPointRow *nearestPoint = new nearestPointRow[npSize];
    nearestPointRow *dev_nearestPoint;

    for (int i = 0; i < npSize; ++i) {
        nearestPoint[i].distance = numeric_limits<double>::max();
    }

    HandleError(cudaMalloc(&dev_learn, learnSize * sizeof(double)));
    HandleError(cudaMalloc(&dev_nearestPoint, npSize * sizeof(nearestPointRow)));

    HandleError(cudaMemcpy(dev_learn, learn, learnSize * sizeof(double), cudaMemcpyHostToDevice));

    cout<<blockSize<<"x"<<numBlocks<<endl;
    for (int i = 0; i < 30; ++i) {
        int op_id = bencher->add_op(std::to_string(i));

        unsigned correct = 0;
        for (auto &testRow: test) {

            if (knn::testRecord(dev_learn, dev_nearestPoint, nearestPoint, testRow, learnSize, learnRows, npSize,
                                npRows, numBlocks, blockSize, tasksPerThread, neighbours) == testRow[3])
                ++correct;
        }

        bencher->end_op(op_id);
        cout << "Accuracy: " << (float) correct / test.size() * 100. << "%" << endl;
        cout << "\nRUN: " << i + 1 << ", TIME: " << bencher->op_timestamp(op_id) << "ms";
    }

//        spit_csv("knn-skin.csv", output, vector < string > {"R", "G", "B", "SKIN"});

    bencher->csv_output((TB_SWITCH == 1 ? "T" : "B") + std::string("_knn") +
                        (TB_SWITCH == 1 ? std ::to_string(THREADS) : std::to_string(BLOCKS)));

    //***************************************************************************************/
    //***************************************************************************************/
    delete[] learn;
    delete[] nearestPoint;
    HandleError(cudaFree(dev_learn));
    HandleError(cudaFree(dev_nearestPoint));

    return 0;
}
