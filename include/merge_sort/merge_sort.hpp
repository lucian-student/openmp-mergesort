#pragma once

#include <vector>

namespace merge_sort {

/// Parallel sort using merge-path partitioning for intra-merge tiles.
template <typename T>
void sort(std::vector<T>& data);

/// Parallel sort using randomized splitters-by-rank (p-path / NI-PDP lecture).
template <typename T>
void sort_by_rank(std::vector<T>& data);

template <typename T>
void sort_serial(std::vector<T>& data);

}  // namespace merge_sort
