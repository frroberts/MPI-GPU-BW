#include <mpi.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <chrono>

int main(int argc, char **argv) {
    
    MPI_Init(&argc, &argv);
    int rank;
    int nprocs;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);


    if (nprocs != 2) 
    {
        if (rank == 0) 
            std::cout << "Error: Run on exantly 2 ranks/GPUs" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (argc==1)
    {
        // print usage
        std::cout << "./a.out devNr1 devNr2 rounds maxSize location \n Where:\n\
         devNr1 is the device number to use on rank 0 \n\
         devNr2 is the device number to use on rank 1 \n\
         rpinds is the number of rounds to run each messge size \n\
         maxSize is the maximum message size to test \n\
         location is either \"host\" or \"device\" to allocate memory on either the host or device \n" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    auto rounds = std::atoi(argv[3]);
    auto maxLen = std::atoi(argv[4]);

    if(rank == 0)
        cudaSetDevice(std::atoi(argv[1]));
    else
        cudaSetDevice(std::atoi(argv[2]));

    int *a;
    int *b;
	std::cout << argv[5] << std::endl;

    if(std::string(argv[5])=="device")
    {
	std::cout << "dev alloc" << std::endl;
	
        cudaMalloc((void**)&a, sizeof(int)*maxLen); 		
        cudaMalloc((void**)&b, sizeof(int)*maxLen); 		
    }
    else
    {
	std::cout << "host alloc" << std::endl;
        a = new int[maxLen];
        b = new int[maxLen];
    }

    std::cout << "start" << std::endl;
    
    for (int len = 1; len <= maxLen; len *= 2) {

        std::chrono::duration<double> best = std::chrono::duration<double>::max();
        
        for (int round = 0; round < rounds; round++) {
        // timer start
            MPI_Barrier(MPI_COMM_WORLD);
            auto start = std::chrono::system_clock::now();
            if(rank == 0)
            {
                // MPI send
                MPI_Send(a, len, MPI_INT, 1, 0, MPI_COMM_WORLD);
            }
            else
            {
                // MPI recv
                MPI_Recv(b, len, MPI_INT, 0, 0, MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE);
            }
            auto end = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = end-start;
            best = std::min(best, elapsed_seconds);
        
        }
        if(rank == 0)
            std::cout << len*sizeof(int) << " BW : " << (len*4.0/1000.0/1000.0/1000.0)/best.count() << std::endl;


    }

    MPI_Finalize();

    return 0;
}
