#include "merge_sort/merge_sort.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace merge_sort
{
  namespace
  {

    template <typename T>
    void serial_merge(const T *left, std::size_t nleft, const T *right, std::size_t nright, T *out)
    {
      std::size_t i = 0;
      std::size_t j = 0;
      std::size_t k = 0;
      while (i < nleft && j < nright)
      {
        if (left[i] <= right[j])
        {
          out[k++] = left[i++];
        }
        else
        {
          out[k++] = right[j++];
        }
      }
      while (i < nleft)
      {
        out[k++] = left[i++];
      }
      while (j < nright)
      {
        out[k++] = right[j++];
      }
    }

    template <typename T>
    void merge_runs(const T *src, T *dst, std::size_t n, std::size_t left_begin, std::size_t width)
    {
      const std::size_t mid = std::min(left_begin + width, n);
      const std::size_t right_end = std::min(left_begin + 2 * width, n);
      const std::size_t nleft = mid - left_begin;
      const std::size_t nright = right_end - mid;

      if (nright == 0)
      {
        std::copy(src + left_begin, src + mid, dst + left_begin);
        return;
      }
      if (nleft == 0)
      {
        std::copy(src + mid, src + right_end, dst + left_begin);
        return;
      }

      serial_merge(src + left_begin, nleft, src + mid, nright, dst + left_begin);
    }

    template <typename T>
    void bottom_up_sort(T *src, T *dst, std::size_t n)
    {
      if (n <= 1)
      {
        return;
      }

      T *a = src;
      T *b = dst;

      for (std::size_t width = 1; width < n; width *= 2)
      {
        const std::size_t num_merges = (n + 2 * width - 1) / (2 * width);
        for (std::size_t m = 0; m < num_merges; ++m)
        {
          const std::size_t left_begin = m * 2 * width;
          merge_runs(a, b, n, left_begin, width);
        }
        std::swap(a, b);
      }

      if (a != src)
      {
        std::copy(a, a + n, src);
      }
    }

  } // namespace

  template <typename T>
  void sort_serial(std::vector<T> &data)
  {
    if (data.size() <= 1)
    {
      return;
    }
    std::vector<T> buffer(data.size());
    bottom_up_sort(data.data(), buffer.data(), data.size());
  }

  template void sort_serial(std::vector<int> &);

} // namespace merge_sort
