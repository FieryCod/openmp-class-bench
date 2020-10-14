#include <iostream>
#include <map>
#include <memory>
#include <omp.h>
#include <vector>
#include <string>
#include <set>
#include <chrono>

#ifndef omp_get_wtime

double omp_get_wtime() {
  return std::chrono::time_point_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now())
      .time_since_epoch()
      .count();
}

#endif

struct Op {
  std::string name;
  double start;
  double end;
  double timestamp;

  Op(std::string name, double start) {
    this->name = name;
    this->start = start;
  }
};

class Bench {
public:
  int add_op(std::string name) {
    ops_names.insert(name);
    int size = state.size();

    state.push_back(Op(name, omp_get_wtime()));

    return size;
  }

  void end_op(int id) {
    double end = omp_get_wtime();

    state[id].end = end;
    state[id].timestamp = end - state[id].start;
  }

  void print_bench() {
    for(auto name : ops_names) {
      print_op_bench(name);
    }
  }

  static void print_benches(std::vector<std::shared_ptr<Bench>> benches) {
    auto ops_names = benches[0]->ops_names;
    std::map<std::string, std::vector<double>> benchs_op_mean;

    for (auto bench : benches) {
      for (auto name : ops_names) {
        std::vector<Op> opv = bench->ops_by_name(name);

        if (benchs_op_mean.find(name) == benchs_op_mean.end()) {
          benchs_op_mean[name] = std::vector<double>();
        }

        benchs_op_mean[name].push_back(mean_op(opv));
      }
    }

    for (auto name : ops_names) {
      print_benches_op_report(name, benchs_op_mean);
    }
  }

  std::vector<Op> ops_by_name(std::string name) {
    std::vector<Op> opv;

    for (auto x : state) {
      if (x.name == name) {
        opv.push_back(x);
      }
    }

    return opv;
  }

private:
  std::vector<Op> state;
  std::set<std::string> ops_names;

  static double mean(std::vector<double> opv) {
    double sum = 0;

    for (auto x : opv) {
      sum += x;
    }

    return sum / opv.size();
  }

  static void print_benches_op_report(std::string name, std::map<std::string, std::vector<double>> op_reps) {
    std::cout << "Mean exec time for benches of op:"
              << " '" << name << "' " << mean(op_reps[name]) << " ms"
              << "\n";
  }

  static double mean_op(std::vector<Op> opv) {
    double sum = 0;

    for (auto x : opv) {
      sum += x.timestamp;
    }

    return sum / opv.size();
  }

  void print_op_bench(std::string name) {
    std::vector<Op> opv = ops_by_name(name);

    std::cout << "Mean exec time:"
              << " '" << name << "' " << mean_op(opv) << " ms" << "\n";
  }
};
