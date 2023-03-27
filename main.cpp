#include "/opt/homebrew/Cellar/mpich/4.1/include/mpi.h"
#include <iostream>
#include <vector>


constexpr int n_1 = 9;
constexpr int n_2 = 4;
constexpr int n_3 = 2;
constexpr int p_1 = 3;
constexpr int p_2 = 2;

void create_matrix(int* matrix, int size_1, int size_2) {
    int k = 1;
    for (int i = 0; i < size_1; ++i) {
        for (int j = 0; j < size_2; ++j) {
            matrix[i * size_1 + j] = k++;
        }
    }
}

void print_matrix(int* matrix, int size_1, int size_2) {
    for (int i = 0; i < size_1; ++i) {
        for (int j = 0; j < size_2; ++j) {
            std::cout << matrix[i * size_1 + j] << ' ';
        }
        std::cout <<  std::endl;
    }
}


using namespace std;
int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int* a = new int[n_1 * n_2];
    int* b = new int[n_2 * n_3];
    if (rank == 0) {
        create_matrix(a, n_1, n_2);
        create_matrix(b, n_2, n_3);
       // print_matrix(a, n_1, n_2);
        //print_matrix(b, n_2, n_3);
    }
    MPI_Comm grid_comm, row_comm, col_comm;
    int dims[2] = {0, 0};
    int periods[2] = {0, 0};
    int coords[2] = {0, 0};
    int reorder = 0;
    int grid_rank;
    int row_rank;
    int col_rank;
    int* row_roots = new int[p_1];
    int row_roots_index = 0;
    MPI_Dims_create(size, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &grid_comm);
    MPI_Comm_rank(grid_comm, &grid_rank);
    MPI_Cart_coords(grid_comm, grid_rank, 2, coords);
    MPI_Comm_rank(grid_comm, &grid_rank);

    MPI_Comm_split(grid_comm, coords[0], rank, &row_comm);
    MPI_Comm_split(grid_comm, coords[1], rank, &col_comm);

    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_rank(col_comm, &col_rank);
    cout << "WORLD RANK: " << rank << ", ROW RANK: " << row_rank << ", COL RANK: " << col_rank << endl;

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&grid_comm);
    MPI_Finalize();
    return 0;
}



