
- 链表
  - [2. 两数相加](https://leetcode.cn/problems/add-two-numbers/)
- 动态规划
  - [3. 无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)
  - [5. 最长回文子串](https://leetcode.cn/problems/longest-palindromic-substring/)
    




[3. 无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        d = dict()
        tmp = 0
        res = 0
        start = 0
        for i, j in enumerate(s):
            if j not in d or (j in d and d[j] < start):
                d[j] = i
                tmp += 1
                res = max(res, tmp )
            else:
                start = d[j] + 1 # 目前的j已经出现过之前的答案里，start需要更新为start位置+1
                tmp = i - d[j] # 长度
                d[j] = i # 更新目前的j的index
                res = max(res, tmp )
            # print(j, tmp, s[start: start + tmp])
        return res 
```

[2. 两数相加](https://leetcode.cn/problems/add-two-numbers/)


```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        surplus = 0 # 进位
        p1, p2 = l1, l2 
        res = ListNode(0)
        head = res 
        while p1 and p2:
            curr = p1.val + p2.val + surplus
            if curr >= 10:
                curr, surplus = curr - 10, 1
            else:
                surplus = 0
            head.next = ListNode(curr)
            head = head.next
            p1, p2 = p1.next, p2.next
        while p1:
            curr = p1.val + surplus 
            if curr >= 10:
                curr, surplus = curr - 10, 1
            else:
                surplus = 0
            head.next = ListNode(curr)
            head = head.next
            p1 = p1.next
        while p2:
            curr = p2.val + surplus 
            if curr >= 10:
                curr, surplus = curr - 10, 1
            else:
                surplus = 0
            head.next = ListNode(curr)
            head = head.next
            p2 = p2.next
        if surplus:
            head.next = ListNode(1)
            head = head.next 
        return res.next
```

[5. 最长回文子串](https://leetcode.cn/problems/longest-palindromic-substring/)


执行用时：
6820 ms
, 在所有 Python3 提交中击败了
32.47%
的用户
内存消耗：
23.7 MB
, 在所有 Python3 提交中击败了
7.30%
的用户

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        res = (0,0)
        length = 0
        n = len(s)
        dp = [[True for _ in range(n)] for _ in range(n)]

        for i in range(n-1, -1, -1):
            for j in range(i+1, n):
                dp[i][j] = s[i]==s[j] and dp[i+1][j-1]
                if dp[i][j] and j - i + 1 > length:
                    res = (i, j)
                    length = j - i + 1
        return s[res[0]: res[1]+1]
```
