#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    // Build K_cat = concat(keys[0..i]) vertically in HBM
    Matrix *k_cat = matrix_memory_allocator.Allocate("k_cat");
    gpu_sim.Copy(keys[0], k_cat, Position::kInGpuHbm);
    for (size_t j = 1; j <= i; ++j) {
      Matrix *k_cat_next = matrix_memory_allocator.Allocate("k_cat_next");
      gpu_sim.Concat(k_cat, keys[j], k_cat_next, /*axis=*/0, Position::kInGpuHbm);
      gpu_sim.ReleaseMatrix(k_cat);
      k_cat = k_cat_next;
    }

    // Build V_cat = concat(values[0..i]) vertically in HBM
    Matrix *v_cat = matrix_memory_allocator.Allocate("v_cat");
    gpu_sim.Copy(values[0], v_cat, Position::kInGpuHbm);
    for (size_t j = 1; j <= i; ++j) {
      Matrix *v_cat_next = matrix_memory_allocator.Allocate("v_cat_next");
      gpu_sim.Concat(v_cat, values[j], v_cat_next, /*axis=*/0, Position::kInGpuHbm);
      gpu_sim.ReleaseMatrix(v_cat);
      v_cat = v_cat_next;
    }

    // Move operands to Shared Memory for computation
    gpu_sim.MoveMatrixToSharedMem(current_query);
    gpu_sim.MoveMatrixToSharedMem(k_cat);
    gpu_sim.MoveMatrixToSharedMem(v_cat);

    // Transpose K_cat in Shared Memory for Q * K_cat^T
    gpu_sim.Transpose(k_cat, Position::kInSharedMemory);

    // S = Q * K_cat^T  => shape (i+1, i+1)
    Matrix *score_mat = matrix_memory_allocator.Allocate("score_mat");
    gpu_sim.MatMul(current_query, k_cat, score_mat);

    // Softmax over rows of score_mat
    Matrix *soft_acc = nullptr;
    for (size_t r = 0; r <= i; ++r) {
      Matrix *row_r = matrix_memory_allocator.Allocate("row_r");
      gpu_sim.GetRow(score_mat, r, row_r, Position::kInSharedMemory);

      Matrix *row_exp = matrix_memory_allocator.Allocate("row_exp");
      gpu_sim.MatExp(row_r, row_exp);

      Matrix *row_sum = matrix_memory_allocator.Allocate("row_sum");
      gpu_sim.Sum(row_exp, row_sum);

      Matrix *row_soft = matrix_memory_allocator.Allocate("row_soft");
      gpu_sim.MatDiv(row_exp, row_sum, row_soft);

      // Accumulate rows vertically into soft_acc
      if (r == 0) {
        soft_acc = matrix_memory_allocator.Allocate("softmax_mat");
        gpu_sim.Copy(row_soft, soft_acc, Position::kInSharedMemory);
      } else {
        Matrix *soft_next = matrix_memory_allocator.Allocate("softmax_next");
        gpu_sim.Concat(soft_acc, row_soft, soft_next, /*axis=*/0,
                       Position::kInSharedMemory);
        gpu_sim.ReleaseMatrix(soft_acc);
        soft_acc = soft_next;
      }

      // Release intermediates for this row
      gpu_sim.ReleaseMatrix(row_r);
      gpu_sim.ReleaseMatrix(row_exp);
      gpu_sim.ReleaseMatrix(row_sum);
      gpu_sim.ReleaseMatrix(row_soft);
    }

    // Result = Softmax * V_cat  => shape (i+1, d)
    Matrix *answer = matrix_memory_allocator.Allocate("answer");
    gpu_sim.MatMul(soft_acc, v_cat, answer);

    // Move answer to HBM for committing
    gpu_sim.MoveMatrixToGpuHbm(answer);

    // Clean up temporaries
    gpu_sim.ReleaseMatrix(k_cat);
    gpu_sim.ReleaseMatrix(v_cat);
    gpu_sim.ReleaseMatrix(score_mat);
    gpu_sim.ReleaseMatrix(soft_acc);

    // Execute the queued instructions
    gpu_sim.Run(false, &matrix_memory_allocator);

    // Commit the answer (Commit after running the simulator.)
    rater.CommitAnswer(*answer);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu