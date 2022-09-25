#include "IMGA.cuh"

int main()
{
	//-------------- Problem variables ----------------
	// number of agents
	int numAgents = 0;
	// number of schedules
	int numSchedules = 0;
	// number of periods
	int numPeriods = 0;
	// length of read_arrE
	int lenArrE = 0;
	// array of count of schedules per Agent
	int *arrASchCount = NULL;
	// array of cummulative count of schedules per Agent
	int *arrAScanSchCount = NULL;
	// length of arrAScanSchCount
	int lenArrL = 0;
	// array of schedules index per Agent => try it on constant memory
	int *arrL = NULL;
	// array of schedules index per Agent host version
	int *read_arrE = NULL;
	// array of schedules index per Agent device version
	int *arrE;
	// array of number of Agents required per period
	int *arrN = NULL;
	// device id
	int deviceId = cudaGetDevice(&deviceId);

	// ----------------- Genetic variables ------------------

	// initilize global variables
	initGblVars(numAgents, numSchedules, numPeriods);
	// initilize arrays
	arrASchCount = (int *)malloc(sizeof(int) * numAgents);
	arrAScanSchCount = (int *)malloc(sizeof(int) * numAgents);

	// read csv data A.csv
	readCSV_A(numAgents, arrASchCount, arrAScanSchCount, lenArrL);
	// read csv data L.csv+
	arrL = (int *)malloc(sizeof(int) * lenArrL);
	readCSV_L(arrL, lenArrL);

	// read csv data E.csv
	lenArrE = numSchedules * numPeriods;
	read_arrE = (int *)malloc(sizeof(int) * lenArrE);
	readCSV_E(read_arrE, lenArrE);
	// check schedule
	// for (int p = 0; p < numPeriods; p++)
	//	printf("period %i: %i\n", p+1, read_arrE[1774*numPeriods + p]);

	// read csv data P.csv
	arrN = (int *)malloc(sizeof(int) * numPeriods);
	readCSV_P(arrN, numPeriods);

	// unified memory for arrE
	cudaMallocManaged(&arrE, lenArrE * sizeof(int));
	// hint to prioritize host transfer
	cudaMemAdvise(arrE, lenArrE * sizeof(int), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
	// read data
	for (int e = 0; e < lenArrE; e++)
	{
		arrE[e] = read_arrE[e];
	}

	// hint for read mostly global data
	cudaMemAdvise(arrE, lenArrE * sizeof(int), cudaMemAdviseSetReadMostly, deviceId);
	// prefetching from host to device
	cudaMemPrefetchAsync(arrE, lenArrE * sizeof(int), deviceId);

	// allocate Device constant memory
	cudaMemcpyToSymbol(const_numAgents, &numAgents, sizeof(int));
	cudaMemcpyToSymbol(const_numSchedules, &numSchedules, sizeof(int));
	cudaMemcpyToSymbol(const_numPeriods, &numPeriods, sizeof(int));
	cudaMemcpyToSymbol(const_lenArrL, &lenArrL, sizeof(int));
	cudaMemcpyToSymbol(const_arrASchCount, arrASchCount, numAgents*sizeof(int));
	cudaMemcpyToSymbol(const_arrAScanSchCount, arrAScanSchCount, numAgents*sizeof(int));
	cudaMemcpyToSymbol(const_arrL, arrL, lenArrL*sizeof(int));
	cudaMemcpyToSymbol(const_arrN, arrN, lenArrE*sizeof(int));





	// deallocate dynamic memory
	free(arrASchCount);
	free(arrAScanSchCount);
	free(arrL);
	free(read_arrE);
	free(arrN);
	cudaFree(arrE);
	// reset device
	cudaDeviceReset();
}

__host__ void initGblVars(int &numAgents, int &numSchedules, int &numPeriods)
{
	// initilize gbl variables
	numAgents = countRows("../MILP/A.csv") - 1;
	printf("\nnum agents: %i", numAgents);
	numSchedules = countRows("../MILP/S.csv") - 1;
	printf("\nnum schedules: %i", numSchedules);
	numPeriods = countRows("../MILP/P.csv") - 1;
	printf("\nnum periods: %i\n", numPeriods);
}

__host__ int countRows(string filePath)
{
	// count rows in a csv file
	int rows = 0;
	ifstream file;
	file.open(filePath);
	string line;
	while (getline(file, line))
		rows++;
	file.close();
	return rows;
}

__host__ void readCSV_A(int &numAgents, int *arrASchCount, int *arrAScanSchCount, int &lenArrL)
{
	// read data A.csv
	string col; // variables from file are here
	// input filename
	string file = "../MILP/A.csv";

	// number of lines
	int i = 0;

	ifstream coeff(file); // opening the file.
	if (coeff.is_open())  // if the file is open
	{
		// ignore first line
		string line;
		getline(coeff, line);

		while (i < numAgents) // while the end of file is NOT reached
		{
			arrAScanSchCount[i] = lenArrL;
			getline(coeff, col, ',');
			getline(coeff, col, ',');
			arrASchCount[i] = stoi(col);
			lenArrL += arrASchCount[i];
			getline(coeff, col, '\n');
			// printf("\nagent %i: %i\n", i, arrASchCount[i]);
			// printf("agent scan %i: %i\n", i, arrAScanSchCount[i]);
			// printf("agent total scan: %i\n", lenArrL);
			i += 1; // increment number of lines
		}
		coeff.close(); // closing the file
	}
	else
		cout << "Unable to open file"; // if the file is not open output
}

__host__ void readCSV_L(int *arrL, int &lenArrL)
{
	// read data A.csv
	string col; // variables from file are here
	// input filename
	string file = "../MILP/L.csv";

	// number of lines
	int i = 0;

	ifstream coeff(file); // opening the file.
	if (coeff.is_open())  // if the file is open
	{
		// ignore first line
		string line;
		getline(coeff, line);

		while (i < lenArrL) // while the end of file is NOT reached
		{
			getline(coeff, col, ',');
			getline(coeff, col, ',');
			getline(coeff, col, ',');
			getline(coeff, col, ',');
			arrL[i] = stoi(col);
			getline(coeff, col, '\n');
			i += 1; // increment number of lines
		}
		coeff.close(); // closing the file
	}
	else
		cout << "Unable to open file"; // if the file is not open output
}

__host__ void readCSV_E(int *read_arrE, int &lenArrE)
{
	// read data A.csv
	string col; // variables from file are here
	// input filename
	string file = "../MILP/E.csv";

	// number of lines
	int i = 0;

	ifstream coeff(file); // opening the file.
	if (coeff.is_open())  // if the file is open
	{
		// ignore first line
		string line;
		getline(coeff, line);

		while (i < lenArrE) // while the end of file is NOT reached
		{
			getline(coeff, col, ',');
			getline(coeff, col, ',');
			getline(coeff, col, ',');
			read_arrE[i] = stoi(col);
			// printf("\nE %i: %i", i, read_arrE[i]);
			getline(coeff, col, '\n');
			i += 1; // increment number of lines
		}
		coeff.close(); // closing the file
	}
	else
		cout << "Unable to open file"; // if the file is not open output
}

__host__ void readCSV_P(int *arrN, int &numPeriods)
{
	// read data A.csv
	string col; // variables from file are here
	// input filename
	string file = "../MILP/P.csv";

	// number of lines
	int i = 0;

	ifstream coeff(file); // opening the file.
	if (coeff.is_open())  // if the file is open
	{
		// ignore first line
		string line;
		getline(coeff, line);

		while (i < numPeriods) // while the end of file is NOT reached
		{
			getline(coeff, col, ',');
			getline(coeff, col, '\n');
			arrN[i] = stoi(col);
			// printf("\nperiod %i: %i", i, arrN[i]);
			i += 1; // increment number of lines
		}
		coeff.close(); // closing the file
	}
	else
		cout << "Unable to open file"; // if the file is not open output
}

__global__ void kernel (){
	
}