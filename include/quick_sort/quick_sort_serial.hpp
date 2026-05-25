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

namespace quick_sort
{
    template <typename T>
    size_t select_pivot(T *data, size_t l, size_t h)
    {
        std::mt19937 random_instance(data[l] + (h - l));
        auto rng = std::bind(std::uniform_int_distribution<size_t>(l, h), random_instance);
        auto pivot_index = rng();
        return pivot_index;
    }

    template <typename T>
    size_t partition(T *data, size_t l, size_t h)
    {
        auto pivot_index = select_pivot(data, l, h);
        auto pivot = data[pivot_index];
        std::swap(data[pivot_index], data[h]);
        ssize_t i = static_cast<ssize_t>(l);
        ssize_t j = static_cast<ssize_t>(h - 1);
        while (i < j)
        {
            if (data[i] > pivot)
            {
                std::swap(data[i], data[j]);
                j--;
            }
            else if (data[j] < pivot)
            {
                std::swap(data[i], data[j]);
                i++;
            }
            else
            {
                i++;
                j--;
            }
        }
        auto new_pivot_index = (i > h) ? h : i;
        if (data[new_pivot_index] < pivot)
        {
            new_pivot_index = new_pivot_index + 1;
            std::swap(data[new_pivot_index], data[h]);
        }
        else if (data[new_pivot_index] > pivot)
            std::swap(data[new_pivot_index], data[h]);
        return new_pivot_index;
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