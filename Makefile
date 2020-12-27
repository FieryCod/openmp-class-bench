PROCESSES=4

resources/dataset.csv:
	mkdir -p resources
	curl https://archive.ics.uci.edu/ml/machine-learning-databases/00196/ConfLongDemo_JSI.txt > resources/dataset.csv

remove_outliers_mpi: src/remove_outliers_mpi.cpp
	@OMPI_CXX=/usr/bin/g++ mpicxx -o remove_outliers src/remove_outliers_mpi.cpp -std=c++17

run_remove_outliers_mpi: remove_outliers_mpi
	@chmod +x remove_outliers_mpi
	@mpirun -np $(PROCESSES) remove_outliers_mpi "resources/skin.csv"
	@rm -Rf remove_outliers_mpi

remove_outliers_omp: src/remove_outliers_omp.cpp
	#mpicxx -o remove_outliers_omp src/remove_outliers_mpi.cpp -std=c++17 -fopenmp
	g++ -fopenmp src/remove_outliers_omp.cpp -o remove_outliers_omp

run_remove_outliers_omp: remove_outliers_omp
	@./remove_outliers_omp "/home/fierycod/skin.csv"
	@rm -Rf remove_outliers_omp
