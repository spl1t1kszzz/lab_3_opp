#include "/opt/homebrew/Cellar/mpich/4.1/include/mpi.h"
#include <iostream>
#include <vector>


constexpr int n_1 = 9;
constexpr int n_2 = 4;
constexpr int n_3 = 4;

std::vector<int> matrix_mult(std::vector<int> &a, std::vector<int> &b, int a_size_1, int b_size_2, int a_size_2) {
    std::vector<int> c(a_size_1 * b_size_2);
    for (int i = 0; i < a_size_1; ++i) {
        for (int j = 0; j < b_size_2; ++j) {
            int sum = 0;
            for (int k = 0; k < a_size_2; ++k) {
                sum += a[i * a_size_2 + k] * b[k * b_size_2 + j];
            }
            c[i * b_size_2 + j] = sum;
        }
    }
    return c;
}


void create_matrix(std::vector<int> &matrix, int size_1, int size_2) {
    for (int i = 0; i < size_1 * size_2; ++i) {
        matrix[i] = i + 1;
    }
}

using namespace std;

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    vector<int> a(n_1 * n_2);
    vector<int> b(n_2 * n_3);
    if (rank == 0) {
        create_matrix(a, n_1, n_2);
        create_matrix(b, n_2, n_3);
        // print_matrix(a, n_1, n_2);
        // print_matrix(b, n_2, n_3);
    }


    MPI_Comm grid_comm, row_comm, col_comm;
    int dims[2] = {0, 0};
    int periods[2] = {0, 0};
    int coords[2] = {0, 0};
    int reorder = 0;
    int grid_rank;
    int row_rank;
    int col_rank;
    // Creating a proc grid
    MPI_Dims_create(size, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &grid_comm);
    MPI_Comm_rank(grid_comm, &grid_rank);
    MPI_Cart_coords(grid_comm, grid_rank, 2, coords);

    // Splitting processes for rows and columns
    MPI_Comm_split(grid_comm, coords[0], rank, &row_comm);
    MPI_Comm_split(grid_comm, coords[1], rank, &col_comm);

    vector<int> a_part(n_1 / dims[0] * n_2);

    if (coords[1] == 0) {
        MPI_Scatter(a.data(), n_1 / dims[0] * n_2, MPI_INT, a_part.data(), n_1 / dims[0] * n_2, MPI_INT, 0, col_comm);
//        for (int i = 0; i < n_1 / dims[0] * n_2; ++i) {
//            std::cout << a_part[i] << ' ';
//        }
//        cout << endl;
        MPI_Barrier(col_comm);

    }
    MPI_Bcast(a_part.data(), n_1 / dims[0] * n_2, MPI_INT, 0, row_comm);


    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_rank(col_comm, &col_rank);
    MPI_Barrier(MPI_COMM_WORLD);
    //cout << "WORLD RANK: " << rank << ", ROW RANK: " << row_rank << ", COL RANK: " << col_rank << endl;

    MPI_Datatype matrix_column, matrix_column_resized;
    MPI_Datatype matrix_result_column, matrix_result_column_resized;

    MPI_Type_vector(n_2, n_3 / dims[1], n_3, MPI_INT, &matrix_column);
    MPI_Type_commit(&matrix_column);
    MPI_Type_create_resized(matrix_column, 0, (int) (n_3 / dims[1] * sizeof(int)), &matrix_column_resized);
    MPI_Type_commit(&matrix_column_resized);

    MPI_Type_vector(n_1 / dims[0], n_3 / dims[1], n_3, MPI_INT, &matrix_result_column);
    MPI_Type_commit(&matrix_result_column);
    MPI_Type_create_resized(matrix_result_column, 0, (int) (n_3 / dims[1] * sizeof(int)),
                            &matrix_result_column_resized);
    MPI_Type_commit(&matrix_result_column_resized);

    vector<int> b_part(n_3 / dims[1] * n_2);
    MPI_Barrier(MPI_COMM_WORLD);
    if (coords[0] == 0) {
        MPI_Scatter(b.data(), 1, matrix_column_resized, b_part.data(), n_3 / dims[1] * n_2, MPI_INT, 0, row_comm);
    }
    MPI_Bcast(b_part.data(), n_3 / dims[1] * n_2, MPI_INT, col_rank, col_comm);
    MPI_Barrier(MPI_COMM_WORLD);
    std::vector<int> c_part = matrix_mult(a_part, b_part, n_1 / dims[0], n_3 / dims[1], n_2);
    std::vector<int> c(n_1 * n_3);
    std::vector<int> recv_counts(size);
    std::vector<int> displs(size);
    std::fill(recv_counts.begin(), recv_counts.end(), 1);
    if (rank == 0) {
        for (int i = 0; i < dims[0]; ++i) {
            for (int j = 0; j < dims[1]; ++j) {
                displs[i * dims[1] + j] = (j * n_3 / dims[1] + i * n_1 / dims[0] * n_3) / (n_3 / dims[1]);
            }
        }
    }


    MPI_Gatherv(c_part.data(), (int) c_part.size(), MPI_INT, c.data(), recv_counts.data(), displs.data(),
                matrix_result_column_resized, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        for (int i = 0; i < n_1; ++i) {
            for (int j = 0; j < n_3; ++j) {
                std::cout << c[i * n_3 + j] << ' ';
            }
            std::cout << std::endl;
        }
    }
    MPI_Type_free(&matrix_column);
    MPI_Type_free(&matrix_column_resized);
    MPI_Type_free(&matrix_result_column);
    MPI_Type_free(&matrix_result_column_resized);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&grid_comm);
    MPI_Comm_free(&col_comm);
    MPI_Finalize();
    return 0;
}


