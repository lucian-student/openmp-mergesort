#pragma once

#include <vector>
#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>
#include <random>
#include <functional>
#include <utility>
#include <iostream>

namespace quick_sort_single
{
    template <typename T>
    size_t select_pivot(T *data, size_t l, size_t h)
    {
        size_t mid = l + (h - l) / 2;

        // Sort the low, middle, and high elements in place to find the median index
        if (data[mid] < data[l])
            std::swap(data[l], data[mid]);
        if (data[h] < data[l])
            std::swap(data[l], data[h]);
        if (data[h] < data[mid])
            std::swap(data[mid], data[h]);

        // Return the index containing the median value
        return mid;
    }


    template <typename T>
    size_t partition(T *data, size_t l, size_t h)
    {
        auto pivot_index = select_pivot(data, l, h);
        auto pivot = data[pivot_index];
        std::swap(data[pivot_index], data[h]);
        size_t i = l;
        for (size_t j = l; j < h; ++j)
        {
            if (data[j] < pivot)
            {
                if (i != j)
                { // Avoid self-swapping
                    std::swap(data[i], data[j]);
                }
                i++;
            }
        }
        std::swap(data[i], data[h]);
        return i;
    }

    template <typename T>
    void quick(T *data, ssize_t l, ssize_t h)
    {
        if (l >= h)
            return;
        size_t mid = partition(data, l, h);
        quick(data, l, mid - 1);
        quick(data, mid + 1, h);
    }

    template <typename T>
    void sort_serial(std::vector<T> &data)
    {
        if (data.size() == 0)
            return;
        T *arr = data.data();
        size_t l = 0;
        size_t h = data.size() - 1;
        quick(arr, l, h);
    }
}