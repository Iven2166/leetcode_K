# 剑指 Offer-leetcode练习题
## 第 9 天 动态规划（中等）
[剑指 Offer 42. 连续子数组的最大和](https://leetcode-cn.com/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/)

输入一个整型数组，数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。
要求时间复杂度为O(n)。

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        dp = 0
        dp_max = -float('inf')
        for i, j in enumerate(nums):
            if j+dp>=j:
                dp = j+dp
                dp_max = max(dp, dp_max)
            else:
                dp = j
                dp_max = max(dp_max, dp)
        return dp_max
```

[剑指 Offer 47. 礼物的最大价值](https://leetcode-cn.com/problems/li-wu-de-zui-da-jie-zhi-lcof/)

在一个 m*n 的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于 0）。你可以从棋盘的左上角开始拿格子里的礼物，并每次向右或者向下移动一格、直到到达棋盘的右下角。给定一个棋盘及其上面的礼物的价值，请计算你最多能拿到多少价值的礼物？
```python
class Solution:
    def maxValue(self, grid: List[List[int]]) -> int:
        # dp[i][j] = max(dp[i-1][j], dp[i][j-1]) + grid[i][j], when i or j is 0 then dealt otherwise
        m, n = len(grid), len(grid[0])
        dp = [[0 for _ in range(n)] for _ in range(m)] # 别写成 dp = [[0]*n]*m]不然会同一个id，导致一个元素变化，三个向量同时变化
        dp[0][0] = grid[0][0] 
        print(dp)
        for i in range(1,n):
            dp[0][i] = dp[0][i-1] + grid[0][i]
        for i in range(1,m):
            dp[i][0] = dp[i-1][0] + grid[i][0]
        for i in range(1,m):
            for j in range(1,n):
                dp[i][j] = max(dp[i][j-1], dp[i-1][j]) + grid[i][j]
        return dp[-1][-1]
```

深拷贝浅拷贝区别：创建dp数组时别用 [[0]*m]*n, 不然不是创建而是浅拷贝。
[说明-1](https://www.runoob.com/w3cnote/python-understanding-dict-copy-shallow-or-deep.html) 、
[说明-2](https://blog.csdn.net/weixin_41888257/article/details/108449289)

## 第 10 天 动态规划（中等）
[剑指 Offer 46. 把数字翻译成字符串](https://leetcode-cn.com/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/)

```python
class Solution:
    def translateNum(self, num: int) -> int:
        # suppose we have n-digit number, then the answer 
        # for i_th digit endding num should be determined 
        # by i-2_th ans & i-1 to i digit num and i-1_th ans & i digit num 
        s1 = set([str(i) for i in range(26)])
        # print(s1)
        num_str = str(num)
        n = len(num_str)
        dp = [0 for i in range(n)]
        for i, j in enumerate(num_str):
            if i == 0:
                dp[0] = 1
            elif i == 1:
                if num_str[0:2] in s1:
                    dp[1] = 2
                else:
                    dp[1] = 1
            # when i > 1
            else:
                if num_str[i-1:i+1] in s1:
                    dp[i] += dp[i-2]
                dp[i] += dp[i-1]
        return dp[-1]
```
和青蛙跳台阶的题目的确是一样的，只不过判断能否加 *dp[-2]* 罢了

