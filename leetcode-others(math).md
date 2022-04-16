
## 479. 最大回文数乘积

给定一个整数 n ，返回 可表示为两个 n 位整数乘积的 最大回文整数 。因为答案可能非常大，所以返回它对 1337 取余 。

[参考](https://leetcode-cn.com/problems/largest-palindrome-product/solution/zui-da-hui-wen-shu-cheng-ji-by-leetcode-rcihq/)

```python
class Solution:
    def largestPalindrome(self, n: int) -> int:
        if n == 1:
            return 9
        upper = 10 ** n - 1 # 左侧顶多就是10^^n-1
        for left in range(upper, upper // 10, -1):  # 枚举回文数的左半部分
            p, x = left, left # p为构造的回文数
            while x:
                p = p * 10 + x % 10  # 翻转左半部分到其自身末尾，构造回文数 p
                x //= 10
            x = upper # x作为其中一个因子，如果p%x==0，那么答案的两个数就存在
            while x * x >= p:
                if p % x == 0:  # x 是 p 的因子
                    return p % 1337
                x -= 1
```


