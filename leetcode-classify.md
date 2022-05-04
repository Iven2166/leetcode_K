```markdown 
-- https://markmap.js.org/repl
# 数据结构
## 字符串
## 栈与队列
## 链表
# 算法
# 动态规划
# 双指针
# 搜索与回溯
# 分治算法
# 位运算
# 排序
```

- 字符串
  
- 数组
    - 双指针
        - [31. 下一个排列 (mid)](https://leetcode-cn.com/problems/next-permutation/)
    - 回溯
        - [78. 子集 (mid)](https://leetcode-cn.com/problems/subsets/)
    - 位运算
        - [78. 子集 (mid)](https://leetcode-cn.com/problems/subsets/)
    - 动态规划
        - [416. 分割等和子集 (mid)](https://leetcode-cn.com/problems/partition-equal-subset-sum/)
- 二叉树
    - 动态规划
        - [96. 不同的二叉搜索树 (mid)](https://leetcode-cn.com/problems/unique-binary-search-trees/)
    


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

[78. 子集 (回溯算法)](https://leetcode-cn.com/problems/subsets/)

给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。

解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。

- 回溯算法（ [参考](https://leetcode-cn.com/problems/subsets/solution/c-zong-jie-liao-hui-su-wen-ti-lei-xing-dai-ni-gao-/) )

> 1.DFS 和回溯算法区别
DFS 是一个劲的往某一个方向搜索，而回溯算法建立在 DFS 基础之上的，但不同的是在搜索过程中，达到结束条件后，恢复状态，回溯上一层，再次搜索。因此回溯算法与 DFS 的区别就是有无状态重置
>
> 2.何时使用回溯算法
当问题需要 "回头"，以此来查找出所有的解的时候，使用回溯算法。即满足结束条件或者发现不是正确路径的时候(走不通)，要撤销选择，回退到上一个状态，继续尝试，直到找出所有解为止
> 
> 作者：show-me-the-code-2
链接：https://leetcode-cn.com/problems/subsets/solution/c-zong-jie-liao-hui-su-wen-ti-lei-xing-dai-ni-gao-/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

执行用时：
36 ms
, 在所有 Python3 提交中击败了
72.79%
的用户
内存消耗：
15.2 MB
, 在所有 Python3 提交中击败了
5.36%
的用户

```python
class Solution:
    def subsets(self, nums):
        tmp, ans = list(), list()

        def dfs(curr, nums):
            if curr == len(nums):
                ans.append(list(tmp)) # 注意不写tmp
                return
            tmp.append(nums[curr])
            dfs(curr + 1, nums)
            tmp.pop()
            dfs(curr + 1, nums)

        dfs(0, nums)
        return ans
```

- 位运算(用01来代表选择的位置) [参考](https://leetcode-cn.com/problems/subsets/solution/zi-ji-by-leetcode-solution/)

执行用时：
40 ms
, 在所有 Python3 提交中击败了
44.82%
的用户
内存消耗：
15.1 MB
, 在所有 Python3 提交中击败了
23.15%
的用户
```python
class Solution: 
    def subsets(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        tmp, ans = list(), list()
        for mask in range(1<<n):
            tmp = list()
            for i in range(n):
                if mask & (1 << i):
                    tmp.append(nums[i])
            ans.append(tmp)
        return ans 
```

[416. 分割等和子集](https://leetcode-cn.com/problems/partition-equal-subset-sum/)

- 动态规划( [参考](https://leetcode-cn.com/problems/partition-equal-subset-sum/solution/fen-ge-deng-he-zi-ji-by-leetcode-solution/) )

```python
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
```


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


