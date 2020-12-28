#include "Bench.h"
#include "rapidcsv.h"
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mpi.h>
#include <string>
#include <vector>


double mean(std::vector<double> xv) {
    double sum = 0;
    #pragma omp parallel for schedule(runtime)
    for (auto x : xv) {
        sum += x;
    }

    return sum / xv.size();
}

double stdf(std::vector<double> xv, double mean_t) {
    double sum = 0;
    #pragma omp parallel for schedule(runtime) default(none) shared(xv) private(mean_t), reduction(+ : sum)
    for (auto x : xv) {
        sum += pow(x - mean_t, 2);
    }

    return sqrt(sum / xv.size());
}

std::vector<double> remove_outliers(std::vector<double> xv, double mean_t, double std_t) {
    std::vector<double> nxv = std::vector<double>();

    #pragma omp parallel
    {
        std::vector<double> nxv_private;

        #pragma omp for schedule(runtime) nowait
        for (auto x : xv) {
            double zscore = (x - mean_t) / std_t;

            if (zscore < 3) {
                nxv_private.push_back(x);
            } else {
                nxv_private.push_back(mean_t);
            }
        }

        #pragma omp critical
        nxv.insert(nxv.end(), nxv_private.begin(), nxv_private.end());
    }

    return nxv;
}

void spit_csv(std::string filename, std::vector <std::vector<double>> ds, std::vector <std::string> cnames) {
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

    std::string file_path = std::string(argv[1]);
    rapidcsv::Document doc(file_path);

    std::vector<double> R = doc.GetColumn<double>("R");
    std::vector<double> G = doc.GetColumn<double>("G");
    std::vector<double> B = doc.GetColumn<double>("B");

    double MEAN_R = mean(R);
    double STD_R = stdf(R, MEAN_R);

    double MEAN_G = mean(G);
    double STD_G = stdf(G, MEAN_G);

    double MEAN_B = mean(B);
    double STD_B = stdf(B, MEAN_B);

    std::unique_ptr <Bench> bencher;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    int COLUMN_SIZE = R.size();
    int COLUMN_PARTITION_SIZE = std::ceil(COLUMN_SIZE / (float) processes);
    int ALL_PARTITION_COLUMN_SIZE = processes * COLUMN_PARTITION_SIZE;

    if (pid == 0) {
        std::cout << "Number of processes running: " << processes;
        std::cout << "\n";
        std::cout << "Column size: " << COLUMN_SIZE << "\n";
        std::cout << "New column size: " << processes * COLUMN_PARTITION_SIZE << "\n";
        bencher = std::make_unique<Bench>(Bench());
    }

    std::vector<double> R_new(ALL_PARTITION_COLUMN_SIZE);
    std::vector<double> G_new(ALL_PARTITION_COLUMN_SIZE);
    std::vector<double> B_new(ALL_PARTITION_COLUMN_SIZE);

    for (int i = 0; i < 30; ++i) {
        int op_id;

        if (pid == 0) {
            op_id = bencher->add_op(std::to_string(i));
        }
        //***************************************************************************************/
        //************************************* Algo ********************************************/
        //***************************************************************************************/

        //************************************** R **********************************************/
        std::vector<double> R_partition(COLUMN_PARTITION_SIZE);
        MPI_Scatter(R.data(), COLUMN_PARTITION_SIZE, MPI_DOUBLE, R_partition.data(), COLUMN_PARTITION_SIZE, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        auto R_partition_M = std::move(remove_outliers(R_partition, MEAN_R, STD_R));
        MPI_Gather(R_partition_M.data(), COLUMN_PARTITION_SIZE, MPI_DOUBLE, R_new.data(), COLUMN_PARTITION_SIZE,
                   MPI_DOUBLE, 0, MPI_COMM_WORLD);
        //***************************************************************************************/

        //************************************** G **********************************************/
        std::vector<double> G_partition(COLUMN_PARTITION_SIZE);
        MPI_Scatter(R.data(), COLUMN_PARTITION_SIZE, MPI_DOUBLE, G_partition.data(), COLUMN_PARTITION_SIZE, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        auto G_partition_M = std::move(remove_outliers(G_partition, MEAN_G, STD_G));
        MPI_Gather(G_partition_M.data(), COLUMN_PARTITION_SIZE, MPI_DOUBLE, G_new.data(), COLUMN_PARTITION_SIZE,
                   MPI_DOUBLE, 0, MPI_COMM_WORLD);
        //***************************************************************************************/

        //************************************** B **********************************************/
        std::vector<double> B_partition(COLUMN_PARTITION_SIZE);
        MPI_Scatter(R.data(), COLUMN_PARTITION_SIZE, MPI_DOUBLE, B_partition.data(), COLUMN_PARTITION_SIZE, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        auto B_partition_M = std::move(remove_outliers(B_partition, MEAN_B, STD_B));
        MPI_Gather(B_partition_M.data(), COLUMN_PARTITION_SIZE, MPI_DOUBLE, B_new.data(), COLUMN_PARTITION_SIZE,
                   MPI_DOUBLE, 0, MPI_COMM_WORLD);
        //***************************************************************************************/

        if (pid == 0) {
            bencher->end_op(op_id);
        }
    }

    if (pid == 0) {
        R_new.resize(COLUMN_SIZE);
        G_new.resize(COLUMN_SIZE);
        B_new.resize(COLUMN_SIZE);

        auto output = std::vector<std::vector<double>> {
                R_new,
                G_new,
                B_new,
                doc.GetColumn<double>("SKIN")
        };

        // spit_csv("remove-outliers-skin.csv", output, std::vector < std::string > {"R", "G", "B", "SKIN"});

        bencher->csv_output("hybrid");
    }

    MPI_Finalize();

    return 0;
}
