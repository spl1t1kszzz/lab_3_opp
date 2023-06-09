#include "mpi.h"
#include <vector>
#include <iostream>
//#include "/usr/local/opt/openblas/include/cblas.h"

constexpr int n_1 = 3360;
constexpr int n_2 = 3360;
constexpr int n_3 = 3360;

std::vector<double> matrix_mult(std::vector<double> &a, std::vector<double> &b, int a_size_1, int b_size_2, int a_size_2) {
    std::vector<double> c(a_size_1 * b_size_2);
    for (int i = 0; i < a_size_1; i++) {
        for (int k = 0; k < a_size_2; k++) {
            for (int j = 0; j < b_size_2; j++) {
                c[i * b_size_2 + j] += a[i * a_size_2 + k] * b[k * b_size_2 + j];
            }
        }
    }
    return c;
}



std::vector<double> create_matrix(int size_1, int size_2) {
    std::vector<double> matrix(size_1 * size_2);
    for (int i = 0; i < size_1; ++i) {
        for (int j = 0; j < size_2; ++j) {
            matrix[i * size_2 + j] = (i == j) ? 2 : 1;
        }
    }
    return matrix;
}

//int blas_check(std::vector<double>& A, std::vector<double>& B, std::vector<double>& my_C, int m, int n, int k) {
//    std::vector<double> blas_C(m * n);
//    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, (const double *)A.data(), n,
//                (const double *)B.data(), n, 0.0, blas_C.data(), n);
//    for (int i = 0; i < m * n; ++i) {
//        if (blas_C[i] != my_C[i]) {
//            return 1;
//        }
//    }
//    return 0;
//}


int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::vector<double> a;
    std::vector<double> b;
    if (rank == 0) {
        a = create_matrix(n_1, n_2);
        b = create_matrix(n_2, n_3);
    }

    MPI_Comm grid_comm, row_comm, col_comm;
    std::vector<int> dims{0, 0};
    std::vector<int> periods{0, 0};
    std::vector<int> coords{0, 0};
    int reorder = 0;
    int grid_rank;
    int row_rank;
    int col_rank;
    const auto start = MPI_Wtime();
    // Creating a proc grid
    MPI_Dims_create(size, 2, dims.data());
    int rows_per_proc = n_1 / dims[0];
    int columns_per_proc = n_3 / dims[1];
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims.data(), periods.data(), reorder, &grid_comm);
    MPI_Comm_rank(grid_comm, &grid_rank);
    MPI_Cart_coords(grid_comm, grid_rank, 2, coords.data());

    // Splitting processes for rows and columns
    MPI_Comm_split(grid_comm, coords[0], rank, &row_comm);
    MPI_Comm_split(grid_comm, coords[1], rank, &col_comm);

    std::vector<double> a_part(rows_per_proc * n_2);

    if (coords[1] == 0) {
        MPI_Scatter(a.data(), rows_per_proc * n_2, MPI_DOUBLE, a_part.data(), rows_per_proc * n_2, MPI_DOUBLE, 0, col_comm);
    }
    MPI_Bcast(a_part.data(), rows_per_proc * n_2, MPI_DOUBLE, 0, row_comm);

    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_rank(col_comm, &col_rank);

    MPI_Datatype matrix_column, matrix_column_resized;
    MPI_Datatype matrix_result_column, matrix_result_column_resized;

    MPI_Type_vector(n_2, columns_per_proc, n_3, MPI_DOUBLE, &matrix_column);
    MPI_Type_commit(&matrix_column);
    MPI_Type_create_resized(matrix_column, 0, (int) (columns_per_proc * sizeof(double)), &matrix_column_resized);
    MPI_Type_commit(&matrix_column_resized);

    MPI_Type_vector(rows_per_proc, columns_per_proc, n_3, MPI_DOUBLE, &matrix_result_column);
    MPI_Type_commit(&matrix_result_column);
    MPI_Type_create_resized(matrix_result_column, 0, (int) (columns_per_proc * sizeof(double )),
                            &matrix_result_column_resized);
    MPI_Type_commit(&matrix_result_column_resized);

    std::vector<double> b_part(columns_per_proc * n_2);
    if (coords[0] == 0) {
        MPI_Scatter(b.data(), 1, matrix_column_resized, b_part.data(), columns_per_proc * n_2, MPI_DOUBLE, 0, row_comm);
    }
    MPI_Bcast(b_part.data(), columns_per_proc * n_2, MPI_DOUBLE, 0, col_comm);
    std::vector<double> c_part = matrix_mult(a_part, b_part, rows_per_proc, columns_per_proc, n_2);
    std::vector<double> c;
    std::vector<int> recv_counts;
    std::vector<int> displs(size);
    if (rank == 0) {
        c.resize(n_1 * n_3);
        recv_counts.resize(size);
        std::fill(recv_counts.begin(), recv_counts.end(), 1);
        for (int i = 0; i < dims[0]; ++i) {
            for (int j = 0; j < dims[1]; ++j) {
                displs[i * dims[1] + j] = (j * columns_per_proc + i * rows_per_proc * n_3) / (columns_per_proc);
            }
        }
    }
    MPI_Gatherv(c_part.data(), (int) c_part.size(), MPI_DOUBLE, c.data(), recv_counts.data(), displs.data(),
                matrix_result_column_resized, 0, MPI_COMM_WORLD);
    const auto end = MPI_Wtime();
    if (rank == 0) {
        std::cout << dims[0] << ' ' << dims[1] << std::endl;
        std::cout << end - start << std::endl;
//        if (0 == blas_check(a,b,c,n_1,n_3,n_2)) {
//            std::cout << "BLAS accepted" << std::endl;
//        }
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

