PROCESSES=4

resources/dataset.csv:
	mkdir -p resources
	curl https://archive.ics.uci.edu/ml/machine-learning-databases/00196/ConfLongDemo_JSI.txt > resources/dataset.csv

remove_outliers: src/remove_outliers.cpp
	@OMPI_CXX=/usr/bin/g++ mpicxx -o remove_outliers src/remove_outliers.cpp -std=c++17

run_remove_outliers: remove_outliers
	@chmod +x remove_outliers
	@mpirun -np $(PROCESSES) remove_outliers "resources/skin.csv"
	@rm -Rf remove_outliers
