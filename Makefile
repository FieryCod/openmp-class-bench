resources/dataset.csv:
	mkdir -p resources
	curl https://archive.ics.uci.edu/ml/machine-learning-databases/00196/ConfLongDemo_JSI.txt > resources/dataset.csv

min_max: src/min_max.cpp
	g++ -fopenmp src/min_max.cpp -o min_max

run_min_max: min_max
	@./min_max 4 "resources/skin.csv"
	@rm -Rf min_max

run_min_max_bench: min_max
	@./min_max 4 "resources/skin.csv" 1
	@rm -Rf min_max

min_max_no_omp: src/min_max.cpp
	g++ src/min_max.cpp -o min_max_no_omp

run_min_max_no_omp: min_max_no_omp
	@./min_max_no_omp 4 "resources/skin.csv" 1
	@rm -Rf min_max_no_omp

run_min_max_no_omp_bench: min_max_no_omp
	@./min_max_no_omp 4 "resources/skin.csv" 1
	@rm -Rf min_max_no_omp

standard_scaler: src/standard_scaler.cpp
	g++ -fopenmp src/standard_scaler.cpp -o standard_scaler

run_standard_scaler: standard_scaler
	@./standard_scaler 4 "resources/skin.csv"
	@rm -Rf standard_scaler

run_standard_scaler_bench: standard_scaler
	@./standard_scaler 4 "resources/skin.csv" 1
	@rm -Rf standard_scaler

standard_scaler_no_omp: src/standard_scaler.cpp
	g++ src/standard_scaler.cpp -o standard_scaler_no_omp

run_standard_scaler_no_omp: standard_scaler_no_omp
	@./standard_scaler_no_omp 4 "resources/skin.csv" 1
	@rm -Rf standard_scaler_no_omp

run_standard_scaler_no_omp_bench: standard_scaler_no_omp
	@./standard_scaler_no_omp 4 "resources/skin.csv" 1
	@rm -Rf standard_scaler_no_omp

knn: src/knn.cpp
	g++ -fopenmp src/knn.cpp -o knn

run_knn: knn
	@./knn 4 "resources/skin.csv"
	@rm -Rf knn

run_knn_bench: knn
	@./knn 4 "resources/skin.csv" 1
	@rm -Rf knn

knn_no_omp: src/knn.cpp
	g++ src/knn.cpp -o knn_no_omp

run_knn_no_omp: knn_no_omp
	@./knn_no_omp 4 "resources/skin.csv" 1
	@rm -Rf knn_no_omp

run_knn_no_omp_bench: knn_no_omp
	@./knn_no_omp 4 "resources/skin.csv" 1
	@rm -Rf knn_no_omp
