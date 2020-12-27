#include "Bench.h"
#include "rapidcsv.h"
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <omp.h>
#include <string>
#include <vector>

class RemoveOutliers {
public:
  RemoveOutliers(rapidcsv::Document doc,
                 std::vector<std::string> cvnames) {
    this->cvnames = cvnames;
    this->doc = doc;
  }

  std::vector<double> remove_outliers(std::vector<double> xv) {
    auto mean_t = mean(xv);
    auto std_t = std(xv, mean_t);

    std::vector<double>nxv = std::vector<double>();

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

  std::vector<std::vector<double>> process() {
    auto v = std::vector<std::vector<double>>{
      remove_outliers(doc.GetColumn<double>("R")),
      remove_outliers(doc.GetColumn<double>("G")),
      remove_outliers(doc.GetColumn<double>("B")),
      doc.GetColumn<double>("SKIN")
    };

    return v;
  }

  void spit_csv(
                std::string filename,
                std::vector<std::vector<double>> ds,
                std::vector<std::string>cnames
                ) {
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

  void process_all_and_spit_output() {
    spit_csv("remove_outliers-skin.csv", process(), cvnames);
  }

private:
  std::vector<std::string> cvnames;
  rapidcsv::Document doc;

  double mean(std::vector<double> xv) {
    double sum = 0;

    #pragma omp parallel for schedule(runtime)
    for (auto x : xv) {
      sum += x;
    }

    return sum / xv.size();
  }

  double std(std::vector<double> xv, double mean_t) {
    double sum = 0;

    #pragma omp parallel for schedule(runtime) default(none) shared(xv) private(mean_t), reduction(+ : sum)
    for (auto x : xv) {
      sum += pow(x - mean_t, 2);
    }

    return sqrt(sum / xv.size());
  }
};

int main(int argc, char *argv[]) {
  std::string file_path = std::string(argv[1]);
  rapidcsv::Document doc(file_path);

  std::cout << "\n";
  std::unique_ptr <Bench> bencher = std::make_unique<Bench>(Bench());

  for (int i = 0; i < 30; ++i) {
    int id = bencher->add_op(std::to_string(i));

    auto algo = std::make_unique<RemoveOutliers>(RemoveOutliers(doc, std::vector<std::string>{"R", "G", "B", "SKIN"}));
    algo->process();
    bencher->end_op(id);
  }

  bencher->csv_output("omp");

  return 0;
}
