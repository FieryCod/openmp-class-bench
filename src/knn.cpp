#include "Bench.h"
#include "rapidcsv.h"
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <omp.h>
#include <string>
#include <vector>

using namespace std;

class Knn {
private:
    vector <array<double,4>> learnData;

    struct nearestPointRow {
        double distance;
        double value;
    };

    double euqlidesDistance(vector<double> testPoint, array<double,4> learnPoint) {
        double sum = 0;
        for (int i = 0; i < testPoint.size() - 1; ++i) {
            sum += pow(testPoint[i] - learnPoint[i], 2);
        }
        return sqrt(sum);
    }
    //TODO add other distance calculations

    void
    checkData(vector <array<double,2>> &nearestPoint, const double distance, const double value, const int neighbours) {
        for (int i = 0; i < neighbours; ++i) {
            if (nearestPoint[i].at(0) > distance) {
                nearestPoint.insert(nearestPoint.begin() + i, {distance, value});
                nearestPoint.pop_back();
                break;
            }
        }
    }

    double getGreatestValue(vector <array<double,2>> &nearestPoint) {
        map<double, int> countMap;

        for (auto &point: nearestPoint) {
            countMap[point[1]]++;
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

public:
    Knn(vector <array<double,4>> learnData) : learnData(learnData) {}

    double testRecord(vector<double> testData, int processes, unsigned short neighbours = 1) {
        vector <array<double,2>> nearestPoint;
        nearestPoint.resize(neighbours);

        for (auto &row: nearestPoint) {
            row[0] = numeric_limits<double>::max();
        }

        int COLUMN_SIZE = learnData.size();
        int COLUMN_PARTITION_SIZE = ceil(COLUMN_SIZE / (float) processes);
        int ALL_PARTITION_COLUMN_SIZE = processes * COLUMN_PARTITION_SIZE;
        nearestPoint.resize(neighbours * processes);

        MPI_Datatype MPI_learn_row;
        MPI_Type_contiguous(4, MPI_DOUBLE, &MPI_learn_row);
        MPI_Type_commit(&MPI_learn_row);

        MPI_Datatype MPI_NEAREST_POINT;
        MPI_Type_contiguous(2, MPI_DOUBLE, &MPI_NEAREST_POINT);
        MPI_Type_commit(&MPI_NEAREST_POINT);

        vector<array<double,4>> LEARN_DATA_partition(COLUMN_PARTITION_SIZE);
        MPI_Scatter(learnData.data(), COLUMN_PARTITION_SIZE, MPI_learn_row, LEARN_DATA_partition.data(),
                    COLUMN_PARTITION_SIZE, MPI_learn_row, 0, MPI_COMM_WORLD);

        auto nearestPoint_M = move(process(learnData, nearestPoint, testData, neighbours));

        MPI_Gather(nearestPoint_M.data(), neighbours, MPI_NEAREST_POINT, nearestPoint.data(), neighbours,
                   MPI_NEAREST_POINT, 0, MPI_COMM_WORLD);

        return getGreatestValue(nearestPoint);;
    }

    vector<array<double,2>> process(vector <array<double,4>> learnData, vector <array<double,2>> nearestPoint,
                           vector<double> testData, unsigned short neighbours = 1) {
        for (auto &learnRow: learnData) {
            checkData(nearestPoint, euqlidesDistance(testData, learnRow), learnRow.at(learnRow.size() - 1), neighbours);
        }

        nearestPoint.resize(neighbours);
        return nearestPoint;
    }

};

void spit_csv(std::string filename, std::vector<std::vector<double>> ds, std::vector<std::string>cnames)
{
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

int main(int argc, char *argv[]) {
    int pid;
    int processes;

    string file_path = string(argv[1]);
//TODO add parameter to chose neighbours
    int neighbours = 5;
    rapidcsv::Document doc(file_path);

    vector<double> R = doc.GetColumn<double>("R");
    vector<double> G = doc.GetColumn<double>("G");
    vector<double> B = doc.GetColumn<double>("B");
    vector<double> SKIN = doc.GetColumn<double>("SKIN");

    vector <array<double,4>> learn;
    vector<vector<double>> test;
    for (int i = 0; i < R.size(); ++i) {
        if (i < R.size() * 0.8) {
            learn.push_back({R[i], G[i], B[i], SKIN[i]});
        } else {
            test.push_back({R[i], G[i], B[i], SKIN[i]});
        }
    }

    unique_ptr <Bench> bencher;
    Knn knn(learn);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    if (pid == 0) {
        cout << "Number of processes running: " << processes;
        cout << "\n";
        bencher = make_unique<Bench>(Bench());
    }

    for (int i = 0; i < 30; ++i) {
        int op_id;

        if (pid == 0) {
            op_id = bencher->add_op(to_string(i));
        }
        //***************************************************************************************/
        //************************************* Algo ********************************************/
        //***************************************************************************************/
        unsigned correct = 0;
        for(auto &testRow: test) {
            if(knn.testRecord(testRow, processes, neighbours) == testRow[3])
                ++correct;
        }
        cout<<"Accuracy: " << (float)correct / test.size() * 100.  << "%" << endl;

        //***************************************************************************************/

        if (pid == 0) {
            bencher->end_op(op_id);
        }
    }

    if (pid == 0) {
//        spit_csv("knn-skin.csv", output, vector < string > {"R", "G", "B", "SKIN"});

        bencher->csv_output(to_string(processes));
    }

    //***************************************************************************************/
    //***************************************************************************************/

    MPI_Finalize();


    return 0;
}
