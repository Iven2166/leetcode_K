
按照"出题指数"刷（https://leetcode.cn/company/bytedance/problemset/）

中等难度

- 链表
    - [2. 两数相加](https://leetcode.cn/problems/add-two-numbers/)
    
- 动态规划
    - [3. 无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)
    - [5. 最长回文子串](https://leetcode.cn/problems/longest-palindromic-substring/)
    
- 双指针
    - [15. 三数之和](https://leetcode.cn/problems/3sum/)
    - [11. 盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/)
    - [33. 搜索旋转排序数组](https://leetcode.cn/problems/search-in-rotated-sorted-array/)

- 回溯法
    - [46. 全排列](https://leetcode.cn/problems/permutations/)
    





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

[15. 三数之和](https://leetcode.cn/problems/3sum/)

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        ans = list()
        n = len(nums)
        if n <= 2:
            return list()
        nums.sort()
        l, r = 0, n - 1
        for first in range(n):
            if first > 0 and nums[first-1] == nums[first]:
                continue # 继续往前找first不同的数
            elif nums[first] > 0: # 加上该句，时间从 44%->80%
                break 
            third = n - 1
            target = - nums[first]
            for second in range(first + 1, n):
                if second > first + 1 and nums[second-1] == nums[second]:
                    continue
                while third > second and nums[third] + nums[second] > target:
                    third -= 1
                if second == third:
                    break
                if nums[second] + nums[third] == target:
                    ans.append([ nums[first], nums[second], nums[third] ])
        return ans 
```

[11. 盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/)


- 解法：反证法：hl < hr时，就算r--，hr变大变小，因为hl是min，所以面积肯定变小，则应该l++

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        ans = 0
        n = len(height)
        l, r = 0, n - 1
        while l < r:
            hl, hr = height[l], height[r]
            h = min(hl, hr )
            ans = max(h * (r - l), ans)
            if hl < hr :
                l += 1
            else:
                r -= 1 
        return ans
```

[46. 全排列](https://leetcode.cn/problems/permutations/)

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        ans = list()
        visit = [0 for _ in range(n)] 
        def dfs(visit, curr):
            nonlocal ans 
            if sum(visit) == n:
                ans.append(list(curr))
                return 
            for i, j in enumerate(visit):
                if j == 0:
                    visit[i] = 1
                    curr.append(nums[i])
                    dfs(visit, curr)
                    visit[i] = 0
                    curr.pop()
        
        dfs(visit, list())
        return ans 
```

[33. 搜索旋转排序数组](https://leetcode.cn/problems/search-in-rotated-sorted-array/)

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        n = len(nums)
        if len(nums)==1:
            return 0 if nums[0]==target else -1
        if len(nums)==2:
            return 0 if nums[0]==target else 1 if nums[1]==target else -1
        l, r = 0, n - 1
        while l < r:
            m = (r + l) >> 1
            a, b, c = nums[l], nums[m], nums[r]
            # print(a, b, c)
            # 如果没有明显的l和r的m+-项，则需要判断a、c是否等于target，且l<r；要么就需要写好m+-1的变更条件，才要l<=r
            if b == target:
                return m 
            if a == target:
                return l 
            if c == target:
                return r
            if c < a and b <= c:
                if target < b or target > a:
                    r = m - 1
                else:
                    l = m
            elif c < a and b >= a:
                if target > b or target < c:
                    l = m + 1
                else:
                    r = m
            elif a < c:
                if target > b:
                    l = m + 1
                else:
                    r = m - 1
        if b == target:
            return m 
        return -1
```