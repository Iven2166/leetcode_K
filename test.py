# -*- coding: UTF-8 -*-

def quick_sort(arr, l, r):
    # 子数组长度为 1 时终止递归
    if l >= r:
        return
    # 哨兵划分操作（以 arr[l] 作为基准数）
    i, j = l, r
    while i < j:
        while i < j and arr[j] >= arr[l]:
            j -= 1
        while i < j and arr[i] <= arr[l]:
            i += 1
        arr[i], arr[j] = arr[j], arr[i]
    print(arr, l, r)
    arr[l], arr[i] = arr[i], arr[l]
    print(arr, l, r)
    # 递归左（右）子数组执行哨兵划分
    quick_sort(arr, l, i - 1)
    quick_sort(arr, i + 1, r)

tmp = [4,3,2,1]
quick_sort(tmp, 0, len(tmp)-1)
print(tmp)
