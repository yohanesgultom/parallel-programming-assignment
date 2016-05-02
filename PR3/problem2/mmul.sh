gcc mmul.c -o mmul.o &&
echo "A_row\tA_col/v_row\tv_col\treps\tT(s)" > mmul.result &&
./mmul.o 10000 10000 1 3 >> mmul.result &&
./mmul.o 15000 15000 1 3 >> mmul.result &&
./mmul.o 20000 20000 1 3 >> mmul.result &&
./mmul.o 25000 25000 1 3 >> mmul.result &&
./mmul.o 30000 30000 1 3 >> mmul.result &&
./mmul.o 35000 35000 1 3 >> mmul.result &&
./mmul.o 40000 40000 1 3 >> mmul.result
