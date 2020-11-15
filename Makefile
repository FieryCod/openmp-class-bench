THREADS=20
BLOCKS=12253
TB_SWITCH=1

resources/dataset.csv:
	mkdir -p resources
	curl https://archive.ics.uci.edu/ml/machine-learning-databases/00196/ConfLongDemo_JSI.txt > resources/dataset.csv

min_max: src/min_max.cu
	nvcc -std=c++17 src/min_max.cu -o min_max

run_min_max: min_max
	@chmod +x min_max
	@./min_max "resources/skin.csv" $(THREADS) $(BLOCKS) $(TB_SWITCH)
	@rm -Rf min_max

standard_scaler: src/standard_scaler.cu
	nvcc -std=c++17 src/standard_scaler.cu -o standard_scaler

run_standard_scaler: standard_scaler
	@chmod +x standard_scaler
	@./standard_scaler "resources/skin.csv" $(THREADS) $(BLOCKS) $(TB_SWITCH)
	@rm -Rf standard_scaler

knn: src/knn.cu
	nvcc -std=c++17 src/knn.cu -o knn

run_knn: knn
	@chmod +x knn
	@./knn "resources/skin1.csv" $(THREADS) $(BLOCKS) $(TB_SWITCH)
	@rm -Rf knn
