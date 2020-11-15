#include "Bench.h"
#include "rapidcsv.h"
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

static void HandleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

void __global__ min_max(double *xv, double *min_t, double *max_t) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  printf("%d\n", index);
  // std::vector<double> nxv = std::vector<double>();

  // for (auto x : xv) {
  //   nxv.push_back((x - min_t) / (max_t - min_t));
  // }

  // return nxv;
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
  int THREADS = 20;

  /**
   * Initialize doc & file path
   */
  std::string file_path = std::string(argv[1]);
  rapidcsv::Document doc(file_path);

  /**
   * Read CSV
   */
  std::vector<double> R = doc.GetColumn<double>("R");
  std::vector<double> G = doc.GetColumn<double>("G");
  std::vector<double> B = doc.GetColumn<double>("B");
  int size = R.size() * sizeof(double);

  /**
   * Calc min, max for columns
   */
  double MIN_R = min(R);
  double MAX_R = max(R);

  double MIN_G = min(G);
  double MAX_G = max(G);

  double MIN_B = min(B);
  double MAX_B = max(B);

  int MIN_BLOCK_SIZE = ceil(R.size() / ((int) THREADS));
  std::cout << "MIN_BLOCK_SIZE: " << MIN_BLOCK_SIZE;

  /**
   * Create global bencher
   */
  std::unique_ptr<Bench> bencher = std::make_unique<Bench>(Bench());
  int op_id = bencher->add_op(std::to_string(1));

  /**
   * Create device vars
   */
  double *d_MIN_R, *d_MAX_R;
  // double *d_MIN_G, *d_MAX_G;
  // double *d_MIN_B, *d_MAX_B;
  double *d_R, *d_G, *d_B;

  /**
   * Alloc device memory
   */
  HANDLE_ERROR(cudaMalloc((void **) &d_R, size));
  HANDLE_ERROR(cudaMalloc((void **) &d_G, size));
  HANDLE_ERROR(cudaMalloc((void **) &d_B, size));

  HANDLE_ERROR(cudaMalloc((void **) &d_MIN_R, sizeof(double)));
  HANDLE_ERROR(cudaMalloc((void **) &d_MAX_R, sizeof(double)));

  /**
   * Copy vars to device memory
   */
  HANDLE_ERROR(cudaMemcpy(d_R, &R[0], size, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_MIN_R, &MIN_R, sizeof(double), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_MAX_R, &MAX_R, sizeof(double), cudaMemcpyHostToDevice));

  min_max<<<MIN_BLOCK_SIZE, THREADS>>>(d_R, d_MIN_R, d_MAX_R);

  /**
   * Copy modified data to host
   */
  HANDLE_ERROR(cudaMemcpy(&R[0], d_R, size, cudaMemcpyDeviceToHost));

  // min_max(d_R, double *min_t, double *max_t)
  bencher->end_op(op_id);





  /*
   * Deallocate memory
   */

  HANDLE_ERROR(cudaFree(d_R)); HANDLE_ERROR(cudaFree(d_G)); HANDLE_ERROR(cudaFree(d_B));
  HANDLE_ERROR(cudaFree(d_MIN_R)); HANDLE_ERROR(cudaFree(d_MAX_R));

  // for (int i = 0; i < 30; ++i) {
  //   int op_id;

  //   if (pid == 0) {
  //   }
  //   //***************************************************************************************/
  //   //************************************* Algo ********************************************/
  //   //***************************************************************************************/

  //   //************************************** R **********************************************/
  //   std::vector<double> R_partition(COLUMN_PARTITION_SIZE);
  //   MPI_Scatter(R.data(), COLUMN_PARTITION_SIZE, MPI_DOUBLE, R_partition.data(), COLUMN_PARTITION_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  //   auto R_partition_M = std::move(min_max(R_partition, MIN_R, MAX_R));
  //   MPI_Gather(R_partition_M.data(), COLUMN_PARTITION_SIZE, MPI_DOUBLE, R_new.data(), COLUMN_PARTITION_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  //   //***************************************************************************************/

  //   //************************************** G **********************************************/
  //   std::vector<double> G_partition(COLUMN_PARTITION_SIZE);
  //   MPI_Scatter(R.data(), COLUMN_PARTITION_SIZE, MPI_DOUBLE, G_partition.data(), COLUMN_PARTITION_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  //   auto G_partition_M = std::move(min_max(G_partition, MIN_G, MAX_G));
  //   MPI_Gather(G_partition_M.data(), COLUMN_PARTITION_SIZE, MPI_DOUBLE, G_new.data(), COLUMN_PARTITION_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  //   //***************************************************************************************/

  //   //************************************** B **********************************************/
  //   std::vector<double> B_partition(COLUMN_PARTITION_SIZE);
  //   MPI_Scatter(R.data(), COLUMN_PARTITION_SIZE, MPI_DOUBLE, B_partition.data(), COLUMN_PARTITION_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  //   auto B_partition_M = std::move(min_max(B_partition, MIN_B, MAX_B));
  //   MPI_Gather(B_partition_M.data(), COLUMN_PARTITION_SIZE, MPI_DOUBLE, B_new.data(), COLUMN_PARTITION_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  //   //***************************************************************************************/

  //   if (pid == 0) {
  //     bencher->end_op(op_id);
  //   }
  // }

  // if (pid == 0) {
  //   R_new.resize(COLUMN_SIZE);
  //   G_new.resize(COLUMN_SIZE);
  //   B_new.resize(COLUMN_SIZE);

  //   auto output = std::vector<std::vector<double>>{
  //     R_new,
  //     G_new,
  //     B_new,
  //     doc.GetColumn<double>("SKIN")
  //   };

  //   spit_csv("min_max-skin.csv", output, std::vector<std::string>{"R", "G", "B", "SKIN"});

  //   bencher->csv_output(std::to_string(processes));
  // }

  // //***************************************************************************************/
  // //***************************************************************************************/

  // MPI_Finalize();


  return 0;
}
