#include "Bench.h"
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

    __global__ void process(double *learnData, unsigned learnDataSize, unsigned learnDataRows,
                            nearestPointRow *nearestPoint, unsigned nearestPointSize, double *testData,
                            unsigned short neighbours = 1) {
        unsigned index = threadIdx.x + blockIdx.x * blockDim.x;

        if (index < learnDataRows) {
            unsigned realIndex = index * learnDataSize / learnDataRows;
            nearestPoint[index].distance =
                    knn::euqlidesDistance(testData, &learnData[realIndex], neighbours);
            nearestPoint[index].value = *(learnData + realIndex + 3);
        }
    }

    __host__ __device__ void getCloseNeighbours(nearestPointRow *nearestPointsInput, nearestPointRow *closeNeighbourList,
                                                unsigned indexStart, unsigned indexEnd, unsigned short neighbours) {
        for (unsigned short i = 0; i < neighbours; ++i) {
            closeNeighbourList[i].distance = numeric_limits<double>::max();
        }

        nearestPointRow temp;
        for (unsigned i = indexStart; i < indexEnd; ++i) {
            if (closeNeighbourList[neighbours - 1].distance > nearestPointsInput[i].distance) {
                closeNeighbourList[neighbours - 1] = nearestPointsInput[i];
                for (unsigned short j = neighbours - 2; j >= 0; --j) {
                    if (closeNeighbourList[j + 1].distance < closeNeighbourList[j].distance) {
                        temp = closeNeighbourList[j + 1];
                        closeNeighbourList[j + 1] = closeNeighbourList[j];
                        closeNeighbourList[j] = temp;
                    } else
                        break;
                }
            }
        }
    }

    double testRecord(double *dev_learnData, nearestPointRow *dev_nearestPointReturn, nearestPointRow *nearestPointInitial,
                      array<double, 4> testData, unsigned dSize, unsigned dRows, unsigned npSize, int blocks, int threads,
                      unsigned short neighbours = 1) {
        double *dev_test;
        HANDLE_ERROR(cudaMalloc(&dev_test, 4 * sizeof(double)));
        HANDLE_ERROR(cudaMemcpy(dev_test, testData.data(), 4 * sizeof(double), cudaMemcpyHostToDevice));

        HANDLE_ERROR(cudaMemcpy(dev_nearestPointReturn, nearestPointInitial, npSize * sizeof(nearestPointRow),
                                cudaMemcpyHostToDevice));

        knn::process<<<blocks, threads>>>(dev_learnData, dSize, dRows, dev_nearestPointReturn, npSize,
                dev_test, neighbours);
        cudaDeviceSynchronize();

        nearestPointRow *nearestPointResult, *nearestPointResult2;
        nearestPointResult2 = new nearestPointRow[neighbours];
        nearestPointResult = new nearestPointRow[npSize];

        HANDLE_ERROR(cudaMemcpy(nearestPointResult, dev_nearestPointReturn, npSize * sizeof(nearestPointRow),
                                cudaMemcpyDeviceToHost));

        getCloseNeighbours(nearestPointResult, nearestPointResult2, 0, npSize, neighbours);

        double res = getGreatestValue(nearestPointResult2, neighbours);
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

    const unsigned npSize = learnRows;
    nearestPointRow *nearestPoint = new nearestPointRow[npSize];
    nearestPointRow *dev_nearestPoint;

    for (int i = 0; i < npSize; ++i) {
        nearestPoint[i].distance = numeric_limits<double>::max();
    }

    HANDLE_ERROR(cudaMalloc(&dev_learn, learnSize * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&dev_nearestPoint, npSize * sizeof(nearestPointRow)));

    HANDLE_ERROR(cudaMemcpy(dev_learn, learn, learnSize * sizeof(double), cudaMemcpyHostToDevice));

    for (int i = 0; i < 30; ++i) {
        int op_id = bencher->add_op(std::to_string(i));

        unsigned correct = 0;
        for (auto &testRow: test) {
            if (knn::testRecord(dev_learn, dev_nearestPoint, nearestPoint, testRow, learnSize, learnRows, npSize,
                                BLOCKS, THREADS, neighbours) == testRow[3])
                ++correct;
        }

        bencher->end_op(op_id);
        cout << "Accuracy: " << (float) correct / test.size() * 100. << "%" << endl;
        cout << "RUN: " << i + 1 << ", TIME: " << bencher->op_timestamp(op_id) << "ms"<<endl<<endl;
    }

//        spit_csv("knn-skin.csv", output, vector < string > {"R", "G", "B", "SKIN"});

    bencher->csv_output((TB_SWITCH == 1 ? "T" : "B") + std::string("_knn") +
                        (TB_SWITCH == 1 ? std::to_string(THREADS) : std::to_string(BLOCKS)));

    delete[] learn;
    delete[] nearestPoint;
    HANDLE_ERROR(cudaFree(dev_learn));
    HANDLE_ERROR(cudaFree(dev_nearestPoint));

    return 0;
}
