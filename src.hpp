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

    // Transpose K_cat in Shared Memory for Q_row * K_cat^T
    gpu_sim.Transpose(k_cat, Position::kInSharedMemory);

    // Build answer row-by-row to reduce peak SRAM usage
    Matrix *answer_acc = nullptr;
    for (size_t r = 0; r <= i; ++r) {
      // q_r: 1 x d
      Matrix *q_r = matrix_memory_allocator.Allocate("q_r");
      gpu_sim.GetRow(current_query, r, q_r, Position::kInSharedMemory);

      // scores_r = q_r * K_cat^T: 1 x (i+1)
      Matrix *scores_r = matrix_memory_allocator.Allocate("scores_r");
      gpu_sim.MatMul(q_r, k_cat, scores_r);

      // softmax(scores_r)
      Matrix *scores_exp = matrix_memory_allocator.Allocate("scores_exp");
      gpu_sim.MatExp(scores_r, scores_exp);
      Matrix *scores_sum = matrix_memory_allocator.Allocate("scores_sum");
      gpu_sim.Sum(scores_exp, scores_sum);
      Matrix *soft_r = matrix_memory_allocator.Allocate("soft_r");
      gpu_sim.MatDiv(scores_exp, scores_sum, soft_r);

      // ans_r = soft_r * V_cat: 1 x d
      Matrix *ans_r = matrix_memory_allocator.Allocate("ans_r");
      gpu_sim.MatMul(soft_r, v_cat, ans_r);

      // accumulate rows into answer_acc
      if (r == 0) {
        answer_acc = matrix_memory_allocator.Allocate("answer_acc");
        gpu_sim.Copy(ans_r, answer_acc, Position::kInSharedMemory);
      } else {
        Matrix *answer_next = matrix_memory_allocator.Allocate("answer_next");
        gpu_sim.Concat(answer_acc, ans_r, answer_next, /*axis=*/0,
                       Position::kInSharedMemory);
        gpu_sim.ReleaseMatrix(answer_acc);
        answer_acc = answer_next;
      }

      // Release per-row temporaries
      gpu_sim.ReleaseMatrix(q_r);
      gpu_sim.ReleaseMatrix(scores_r);
      gpu_sim.ReleaseMatrix(scores_exp);
      gpu_sim.ReleaseMatrix(scores_sum);
      gpu_sim.ReleaseMatrix(soft_r);
      gpu_sim.ReleaseMatrix(ans_r);
    }

    // Move final answer to HBM for committing
    Matrix *answer = answer_acc;
    gpu_sim.MoveMatrixToGpuHbm(answer);

    // Clean up temporaries
    gpu_sim.ReleaseMatrix(k_cat);
    gpu_sim.ReleaseMatrix(v_cat);
    // k_cat currently is transposed view; release temps

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