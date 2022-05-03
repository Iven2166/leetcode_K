

## 数组
### 中等
[215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)

给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。

```python
import heapq # 小顶堆
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        stack = list()
        for i in nums:
            heapq.heappush(stack, -i)
        while k:
            ans = - heappop(stack)
            k -= 1
        return ans
```

[31. 下一个排列](https://leetcode-cn.com/problems/next-permutation/)

[参考](https://leetcode-cn.com/problems/next-permutation/solution/xia-yi-ge-pai-lie-by-leetcode-solution/)

双指针

```python
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        i = len(nums) - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        if i >= 0:
            j = len(nums) - 1
            while j >= 0 and nums[i] >= nums[j]:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]
        
        left, right = i + 1, len(nums) - 1
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1
```

[128. 最长连续序列](https://leetcode-cn.com/problems/longest-consecutive-sequence/)

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        nums = set(nums)
        ans, max_ans = 0, 0
        for i in nums:
            if i - 1 not in nums:
                curr = i 
                ans = 1 
                while curr + 1 in nums:
                    curr = curr + 1
                    ans += 1
                max_ans = max(max_ans, ans)
        return max_ans
```

[78. 子集](https://leetcode-cn.com/problems/subsets/)

给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。

解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。





## 二叉树

### 中等
[96. 不同的二叉搜索树](https://leetcode-cn.com/problems/unique-binary-search-trees/)

```python
class Solution:
    def numTrees(self, n: int) -> int:
        if n <= 2:
            return n 
        dp = dict()
        dp[0], dp[1], dp[2] = 1, 1, 2
        for i in range(3, n + 1):
            dp[i] = 0
            for j in range(i):
                dp[i] = dp[i] + dp[j] * dp[i - j - 1]
                # print(i, j, dp)
        return dp[n]
```


