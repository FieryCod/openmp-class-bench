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

standard_scaller: src/standard_scaler.cpp
	@OMPI_CXX=/usr/bin/g++ mpicxx -o standard_scaller src/standard_scaler.cpp -std=c++17

run_standard_scaller: standard_scaller
	@chmod +x standard_scaller
	@mpirun -np $(PROCESSES) standard_scaller "resources/skin.csv"
	@rm -Rf standard_scaller
