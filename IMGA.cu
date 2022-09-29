#include "IMGA.cuh"

int main()
{
	cudaError_t err = cudaSuccess;
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
	// curand state
	curandState *d_state;

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
	printf("len L: %i", lenArrL);
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
	err = cudaMallocManaged(&arrE, lenArrE * sizeof(int));

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector arrE (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// hint to prioritize host transfer
	cudaMemAdvise(arrE, lenArrE * sizeof(int), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
	// read data
	for (int e = 0; e < lenArrE; e++)
	{
		arrE[e] = read_arrE[e];
	}

	// curand memory allocation
	err = cudaMallocManaged(&d_state, BLOCKS_PER_GRID * THREADS_PER_BLOCK * sizeof(curandState));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector d_state (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	// launch init kernel
	cudaMemAdvise(d_state, BLOCKS_PER_GRID * THREADS_PER_BLOCK * sizeof(curandState), cudaMemAdviseSetPreferredLocation, deviceId);
	setup_curand<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(d_state);
	cudaDeviceSynchronize();

	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch setup_curand kernel (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	//------- allocate Device constant memory-----
	err = cudaMemcpyToSymbol(const_numSchedules, &numSchedules, sizeof(int));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device constant const_numSchedules (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpyToSymbol(const_numPeriods, &numPeriods, sizeof(int));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device constant const_numPeriods (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpyToSymbol(const_lenArrL, &lenArrL, sizeof(int));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device constant const_lenArrL (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	//------- allocate Device constant arrays memory-----
	err = cudaMemcpyToSymbol(const_arrASchCount, arrASchCount, numAgents * sizeof(int));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device constant const_arrASchCount (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpyToSymbol(const_arrAScanSchCount, arrAScanSchCount, numAgents * sizeof(int));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device constant const_arrAScanSchCount (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpyToSymbol(const_arrL, arrL, lenArrL * sizeof(int));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device constant const_arrL (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpyToSymbol(const_arrN, arrN, numPeriods * sizeof(int));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device constant const_arrN (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	//-----hints------ unified memory
	// hint for read mostly global data
	cudaMemAdvise(arrE, lenArrE * sizeof(int), cudaMemAdviseSetReadMostly, deviceId);
	// prefetching from host to device
	cudaMemPrefetchAsync(arrE, lenArrE * sizeof(int), deviceId);
	cudaMemAdvise(d_state, BLOCKS_PER_GRID * THREADS_PER_BLOCK * sizeof(curandState), cudaMemAdviseSetReadMostly, deviceId);

	// execute kernel
	printf("\nblocks: %i", BLOCKS_PER_GRID);
	printf("\nthreads: %i", THREADS_PER_BLOCK);
	size_t shared_bytes = SUB_POPULATION_SIZE * numAgents * sizeof(int);
	printf("\nshared_bytes: %zu bytes", shared_bytes);
	kernel_IMGA<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK, shared_bytes>>>(arrE, d_state);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch kernel_IMGA kernel (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	// deallocate dynamic memory
	free(arrASchCount);
	free(arrAScanSchCount);
	free(arrL);
	free(read_arrE);
	free(arrN);
	cudaFree(arrE);
	cudaFree(d_state);
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

__global__ void setup_curand(curandState *state)
{
	// each island has a different seed, and each individual has a different sequence
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	curand_init(blockIdx.x, threadIdx.x, 0, &state[tid]);
}

__global__ void kernel_IMGA(int *arrE, curandState *state)
{
	// initlize cooperative groups
	// the grid represents the global population
	cg::grid_group grid = cg::this_grid();
	// each block represents an island population
	cg::thread_block block = cg::this_thread_block();
	// each tile represents an individual
	cg::thread_block_tile<THREADS_PER_INDIVIDUAL> tile_individual = cg::tiled_partition<THREADS_PER_INDIVIDUAL>(block);

	// shared memory
	// island population of parents
	extern int __shared__ subPopulation[];
	// island population of children
	extern int __shared__ subOffsprings[];
	// fitnes vector for each island
	int __shared__ arrFitness
		[SUB_POPULATION_SIZE];
	// Parent ID vector for each island
	int __shared__ arrParents[SUB_POPULATION_SIZE];

	// ------------------- Initilize sub-populations ------------------------------
	// Copy random number state to local memory (registers) for efficiency
	curandState localState = state[grid.thread_rank()];

	for (int a = tile_individual.thread_rank(); a < AGENTS_SIZE; a += tile_individual.size())
	{
		float random_value = curand_uniform(&localState) * const_arrASchCount[a];
		int random_pos = (int)truncf(random_value);

		subPopulation[tile_individual.meta_group_rank() * AGENTS_SIZE + a] = const_arrL[const_arrAScanSchCount[a] + random_pos];
		//--------Validate initial Population
		// int idb = blockIdx.x;
		// if (idb == 0)
		// {
		// 	printf("\nblock: %i, individual: %i, agent: %i, feasible: %i, startID: %i, random: %i, scheduleID: %i", block.group_index().x, tile_individual.meta_group_rank(), a, const_arrASchCount[a], const_arrAScanSchCount[a], random_pos, const_arrL[const_arrAScanSchCount[a] + random_pos]);
		// }
	}

	//---------------------start epoch-----------------------------
	//---------------------start generation------------------------
	// initilize fitness and parent vectors for each generation
	if (tile_individual.thread_rank() == 0)
	{
		arrFitness[tile_individual.meta_group_rank()] = 0;
		arrParents[tile_individual.meta_group_rank()] = 0;
	}
	// syncronize all threads from the same island
	cg::sync(block);
	//------------------ calculate fitness--------------------------
	// local memory
	int objective = 0;
	int active_agents = 0;
	for (int p = 0; p < const_numPeriods; p++)
	{
		active_agents = 0;
		// grid stride loops along agents dimension
		for (int a = tile_individual.thread_rank(); a < AGENTS_SIZE; a += tile_individual.size())
		{
			int idSchedule = subPopulation[tile_individual.meta_group_rank() * AGENTS_SIZE + a];
			active_agents += arrE[idSchedule * const_numPeriods + p];
			//  print schedules and set covering for period p
			// if (block.group_index().x == 0 && tile_individual.meta_group_rank() == 0)
			// {
			// 	printf("\nagent %i, schedule %i: %i", a, idSchedule, arrE[idSchedule * const_numPeriods + p]);
			// }
		}

		cg::sync(tile_individual);
		// reduce cooperative function
		active_agents = cg::reduce(tile_individual, active_agents, cg::plus<int>());

		// calculate objective funtion
		if (tile_individual.thread_rank() == 0)
		{
			// objective could be moved to shared memory
			objective = objective + max(const_arrN[p] - active_agents, 0);
			// print fo along the periods
			//  if (block.group_index().x == 0 && tile_individual.meta_group_rank() == 2)
			//  {
			//  	printf("\nPeriodo %i, Activos: %i, requeridos: %i, fo: %i", p, active_agents, const_arrN[p], objective);
			//  }
			// roulette selection
			// atomicAdd(&totalFitness[block.group_index().x],objective);
		}
	}
	if (tile_individual.thread_rank() == 0)
	{
		arrFitness[tile_individual.meta_group_rank()] = objective;
	}
	// print fitness vector for island 0
	// if (block.group_index().x == 27 && tile_individual.thread_rank() == 0)
	// {
	// 	printf("\nindividual %i: %i faltantes, %i", tile_individual.meta_group_rank(), objective,arrFitness[tile_individual.meta_group_rank()]);
	// }
	//---------------- tournament selection --------------------
	cg::sync(block);
	int parentID = 0;
	float random_value = curand_uniform(&localState) * SUB_POPULATION_SIZE;
	parentID = (int)truncf(random_value);
	objective = arrFitness[parentID];
	cg::sync(tile_individual);
	// get winner fitness by reduce cooperative function
	//  if (block.group_index().x == 27 && tile_individual.meta_group_rank() == 0)
	//  {
	//   	printf("\nindividual %i: %i faltantes", tile_individual.thread_rank(), objective);
	//  }
	objective = cg::reduce(tile_individual, objective, cg::less<int>());
	// if (block.group_index().x == 0 && tile_individual.meta_group_rank() == 0)
	// {
	//   	printf("\nindividual %i, parentID %i, fitness %i, minimo %i", tile_individual.thread_rank(), parentID, arrFitness[parentID], tile_individual.shfl(objective,0));
	// }
	// deterministic using atomic operators
	if (tile_individual.shfl(objective, 0) == arrFitness[parentID])
	{
		atomicExch(&arrParents[tile_individual.meta_group_rank()], parentID);
	}
	cg::sync(block);
	// if (block.group_index().x == 0 && tile_individual.meta_group_rank() == 0)
	// {
	//   	printf("\nparentID selected %i", arrParents[tile_individual.meta_group_rank()]);
	// }
	//-----------------------Crossover & mutation -------------------------
	// generate crossover point
	if (tile_individual.thread_rank() == 0)
	{
		random_value = curand_uniform(&localState) * AGENTS_SIZE;
		// reuse parentID as crossover point
		parentID = (int)truncf(random_value);
	}

	// first half from parent
	for (int a = tile_individual.thread_rank(); a < tile_individual.shfl(parentID, 0); a += tile_individual.size())
	{
		subOffsprings[tile_individual.meta_group_rank() * AGENTS_SIZE + a] = subPopulation[arrParents[tile_individual.meta_group_rank()] * AGENTS_SIZE + a];
	}
	// second half from individual
	for (int a = tile_individual.shfl(parentID, 0); a < AGENTS_SIZE; a += tile_individual.size())
	{
		subOffsprings[tile_individual.meta_group_rank() * AGENTS_SIZE + a] = subPopulation[tile_individual.meta_group_rank() * AGENTS_SIZE + a];
	}
	cg::sync(block);

	
	//-----------------------Elitism------------------------------
}