PROCESSES=1

resources/dataset.csv:
	mkdir -p resources
	curl https://archive.ics.uci.edu/ml/machine-learning-databases/00196/ConfLongDemo_JSI.txt > resources/dataset.csv

min_max: src/min_max.cpp
	@OMPI_CXX=/usr/bin/g++ mpicxx -o min_max src/min_max.cpp -std=c++17

run_min_max: min_max
	@chmod +x min_max
	@mpirun -np $(PROCESSES) min_max "resources/skin.csv"
	@rm -Rf min_max

standard_scaler: src/standard_scaler.cpp
	@OMPI_CXX=/usr/bin/g++ mpicxx -o standard_scaler src/standard_scaler.cpp -std=c++17

run_standard_scaler: standard_scaler
	@chmod +x standard_scaler
	@mpirun -np $(PROCESSES) standard_scaler "resources/skin.csv"
	@rm -Rf standard_scaler

knn: src/knn.cpp
	@OMPI_CXX=/usr/bin/g++ mpicxx -o knn src/knn.cpp -std=c++17

run_knn: knn
	@chmod +x knn
	@mpirun -np $(PROCESSES) knn "resources/normalized.csv"
	@rm -Rf knn
