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
#
tmp1 = [4,3,2,1]
# quick_sort(tmp, 0, len(tmp)-1)
# print(tmp)

class Solution:
    def subsets(self, nums):
        tmp, ans = list(), list()

        def dfs(curr, nums):
            if curr == len(nums):
                ans.append(list(tmp))
                return
            tmp.append(nums[curr])
            # print(tmp)
            dfs(curr + 1, nums)
            tmp.pop()
            dfs(curr + 1, nums)

        dfs(0, nums)
        return ans

# print(tmp1)
# print(Solution().subsets(tmp1))

class Solution:
    def canPartition(self, nums):
        n = len(nums)
        if n < 2:
            return False
        total = sum(nums)
        maxNum = max(nums)
        if total & 1:
            return False
        target = total // 2
        if maxNum > target:
            return False

        dp = [ [False for _ in range(target+1)] for _ in range(n)]
        for i in range(n):
            dp[i][0] = True
        dp[0][nums[0]] = True

        for i in range(1, n):
            num = nums[i]
            for j in range(1, target + 1):
                if j >= num: # 才有是否需要提取的判断
                    dp[i][j] = dp[i - 1][j] | dp[i - 1][j - num]
                else:
                    dp[i][j] = dp[i - 1][j]

        return dp[n - 1][target]




print(Solution().canPartition(tmp1))