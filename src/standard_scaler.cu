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

void __global__ standard_scaler(double *xv, double std, double mean, int count) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  if (index < count) {
    xv[index] = (xv[index] - mean) / std;
  }
}

double mean(std::vector<double> xv) {
    double sum = 0;
    for (auto x : xv) {
        sum += x;
    }

    return sum / xv.size();
}

double stdf(std::vector<double> xv, double mean_t) {
    double sum = 0;
    for (auto x : xv) {
        sum += pow(x - mean_t, 2);
    }

    return sqrt(sum / xv.size());
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
  /**
   * Initialize doc & file path
   */
  std::string file_path = std::string(argv[1]);
  rapidcsv::Document doc(file_path);
  int THREADS = std::stoi(argv[2]);
  int BLOCKS = std::stoi(argv[3]);
  int TB_SWITCH = std::stoi(argv[4]);

  /**
   * Read CSV
   */
  std::vector<double> R = doc.GetColumn<double>("R");
  std::vector<double> G = doc.GetColumn<double>("G");
  std::vector<double> B = doc.GetColumn<double>("B");
  int count = R.size();
  int size = count * sizeof(double);

  /**
   * Calc MEAN and std for columns
   */
   double MEAN_R = mean(R);
   double STD_R = stdf(R, MEAN_R);

   double MEAN_G = mean(G);
   double STD_G = stdf(G, MEAN_G);

   double MEAN_B = mean(B);
   double STD_B = stdf(B, MEAN_B);

  std::cout << "BLOCKS: " << BLOCKS
            << "\nTHREADS: " << THREADS
            << "\nCOUNT: " << THREADS * BLOCKS
            << "\n\n---------------------------------------\n";

  /**
   * Create global bencher
   */
  std::unique_ptr<Bench> bencher = std::make_unique<Bench>(Bench());

  for (int i = 0; i < 30; ++i) {
    int op_id = bencher->add_op(std::to_string(i));

    /**
     * Create device vars
     */
    double *d_R, *d_G, *d_B;

    /**
     * Alloc device memory
     */
    HANDLE_ERROR(cudaMalloc((void **)&d_R, size));
    HANDLE_ERROR(cudaMalloc((void **)&d_G, size));
    HANDLE_ERROR(cudaMalloc((void **)&d_B, size));

    /**
     * Copy vars to device memory
     */
    HANDLE_ERROR(cudaMemcpy(d_R, &R[0], size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_G, &G[0], size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_B, &B[0], size, cudaMemcpyHostToDevice));

    /**
     * Algorithm
     */
    standard_scaler<<<BLOCKS, THREADS>>>(d_R, STD_R, MEAN_R, count);
    standard_scaler<<<BLOCKS, THREADS>>>(d_G, STD_G, MEAN_G, count);
    standard_scaler<<<BLOCKS, THREADS>>>(d_B, STD_B, MEAN_B, count);

    /**
     * Copy modified data to host
     */
    HANDLE_ERROR(cudaMemcpy(&R[0], d_R, size, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(&G[0], d_G, size, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(&B[0], d_B, size, cudaMemcpyDeviceToHost));

    bencher->end_op(op_id);

    std::cout << "\nRUN: " << i + 1 << ", TIME: " << bencher->op_timestamp(op_id) << "ms";
    /*
     * Deallocate memory
     */
    HANDLE_ERROR(cudaFree(d_R));
    HANDLE_ERROR(cudaFree(d_G));
    HANDLE_ERROR(cudaFree(d_B));
  }

  auto output = std::vector<std::vector<double>>{
    R,
    G,
    B,
    doc.GetColumn<double>("SKIN")
  };

  spit_csv("standard_scaler-skin.csv", output, std::vector<std::string>{"R", "G", "B", "SKIN"});
  bencher->csv_output((TB_SWITCH == 1 ? "T" : "B") + std::string("_standard_scaler") +
                      (TB_SWITCH == 1 ? std ::to_string(THREADS) : std::to_string(BLOCKS)));

  return 0;
}
