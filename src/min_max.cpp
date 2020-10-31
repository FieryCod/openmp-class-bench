#include <cmath>
#include <cstddef>
#include "Bench.h"
#include "rapidcsv.h"
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <cstdio>
#include <mpi.h>
#include <string>
#include <vector>

std::vector<double> min_max(std::vector<double> xv, double min_t, double max_t) {
  std::vector<double> nxv = std::vector<double>();

  for (auto x : xv) {
    nxv.push_back((x - min_t) / (max_t - min_t));
  }

  return nxv;
}

double max(std::vector<double> xv) {
  double max = std::numeric_limits<double>::min();

  for (auto x : xv) {
    if (x > max) {
      max = x;
    }
  }

  return max;
}

double min(std::vector<double> xv) {
  double min = std::numeric_limits<double>::max();

  for (auto x : xv) {
    if (x < min) {
      min = x;
    }
  }

  return min;
}



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

  std::string file_path = std::string(argv[1]);
  rapidcsv::Document doc(file_path);

  std::vector<double> R = doc.GetColumn<double>("R");
  std::vector<double> G = doc.GetColumn<double>("G");
  std::vector<double> B = doc.GetColumn<double>("B");

  double MIN_R = min(R);
  double MAX_R = max(R);

  double MIN_G = min(G);
  double MAX_G = max(G);

  double MIN_B = min(B);
  double MAX_B = max(B);

  std::unique_ptr<Bench> bencher;

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
    MPI_Scatter(R.data(), COLUMN_PARTITION_SIZE, MPI_DOUBLE, R_partition.data(), COLUMN_PARTITION_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    auto R_partition_M = std::move(min_max(R_partition, MIN_R, MAX_R));
    MPI_Gather(R_partition_M.data(), COLUMN_PARTITION_SIZE, MPI_DOUBLE, R_new.data(), COLUMN_PARTITION_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //***************************************************************************************/

    //************************************** G **********************************************/
    std::vector<double> G_partition(COLUMN_PARTITION_SIZE);
    MPI_Scatter(R.data(), COLUMN_PARTITION_SIZE, MPI_DOUBLE, G_partition.data(), COLUMN_PARTITION_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    auto G_partition_M = std::move(min_max(G_partition, MIN_G, MAX_G));
    MPI_Gather(G_partition_M.data(), COLUMN_PARTITION_SIZE, MPI_DOUBLE, G_new.data(), COLUMN_PARTITION_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //***************************************************************************************/

    //************************************** B **********************************************/
    std::vector<double> B_partition(COLUMN_PARTITION_SIZE);
    MPI_Scatter(R.data(), COLUMN_PARTITION_SIZE, MPI_DOUBLE, B_partition.data(), COLUMN_PARTITION_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    auto B_partition_M = std::move(min_max(B_partition, MIN_B, MAX_B));
    MPI_Gather(B_partition_M.data(), COLUMN_PARTITION_SIZE, MPI_DOUBLE, B_new.data(), COLUMN_PARTITION_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //***************************************************************************************/

    if (pid == 0) {
      bencher->end_op(op_id);
    }
  }

  if (pid == 0) {
    R_new.resize(COLUMN_SIZE);
    G_new.resize(COLUMN_SIZE);
    B_new.resize(COLUMN_SIZE);

    auto output = std::vector<std::vector<double>>{
      R_new,
      G_new,
      B_new,
      doc.GetColumn<double>("SKIN")
    };

    spit_csv("min_max-skin.csv", output, std::vector<std::string>{"R", "G", "B", "SKIN"});

    bencher->csv_output(std::to_string(processes));
  }

  //***************************************************************************************/
  //***************************************************************************************/

  MPI_Finalize();


  return 0;
}
