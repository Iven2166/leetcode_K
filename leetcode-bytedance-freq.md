
按照"出题指数"刷（https://leetcode.cn/company/bytedance/problemset/）


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


[56. 合并区间](https://leetcode.cn/problems/merge-intervals/)

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        ans = list()
        intervals.sort(key = lambda x: x[0])
        for i, j in enumerate(intervals):
            if len(ans)==0 or j[0] > ans[-1][1]:
                ans.append(j)
            else:
                ans[-1][1] = max(ans[-1][1],j[1])
        return ans 
```

[31. 下一个排列](https://leetcode.cn/problems/next-permutation/)

- [参考](https://leetcode.cn/problems/next-permutation/solution/xia-yi-ge-pai-lie-by-leetcode-solution/)

```python
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        i = n - 2
        while i >= 0 and nums[i] >= nums[i+1]:
            i -= 1
        if i >= 0:
            j = len(nums) - 1
            while j >= 0 and nums[i] >= nums[j]:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]
        
        i, j = i + 1, len(nums) - 1
        while i < j:
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
            j -= 1
```

[200. 岛屿数量](https://leetcode.cn/problems/number-of-islands/)

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        
        def dfs(last_dir, curr_dir, i, j):
            # nonlocal self.visited
            if (last_dir == 'down' and curr_dir == 'up') or (last_dir == 'up' and curr_dir == 'down'):
                return 
            if (last_dir == 'right' and curr_dir == 'left') or (last_dir == 'left' and curr_dir == 'right'):
                return 
            # last_dir 为上次的方向
            if i >= m or j >= n or i < 0 or j < 0:
                return 
            if grid[i][j] == "1":
                if self.visited[i][j] == 1:
                    return 
                elif self.visited[i][j] == 0:
                    self.visited[i][j] = 1
            
            if grid[i][j] == "0":
                return 
            
            dfs(curr_dir, 'down', i+1, j)
            dfs(curr_dir, 'up', i-1, j)
            dfs(curr_dir, 'right', i, j+1)
            dfs(curr_dir, 'left', i, j-1)
        
        m, n = len(grid), len(grid[0])
        self.visited = [[0 for _ in range(n)] for _ in range(m)]
        
        res = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j]=='1' and self.visited[i][j]==0:
                    dfs(None, None, i, j)
                    res += 1
        return res 
```

- 直接修改grid[i][j]使得其变成0（水）

```python
class Solution:
    def dfs(self, grid, x, y):
        grid[x][y] = '0'
        nr, nc = len(grid), len(grid[0])
        for i,j in [(x-1,y), (x+1,y), (x, y-1), (x, y+1)]:
            if 0<=i<nr and 0<=j<nc and grid[i][j]=="1":
                self.dfs(grid, i, j)

    def numIslands(self, grid: List[List[str]]) -> int:
        nr = len(grid)
        if nr==0: return 0
        nc = len(grid[0])
        num = 0
        for r in range(nr):
            for c in range(nc):
                if grid[r][c]=="1":
                    num += 1
                    self.dfs(grid, r, c)
        return num
```

[22. 括号生成](https://leetcode.cn/problems/generate-parentheses/)

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        def dfs(l, k): 
            nonlocal res 
            # l 左括号数量 r 右括号数量 k 目前的list
            if l < 0:
                return 
            if len(k) == n * 2:
                if l == 0:
                    res.append(''.join(k))
                return     
            if l == 0:
                k.append("(")
                dfs(l + 1, k)
                k.pop()
            elif l > 0:
                k.append("(")
                dfs(l + 1, k)
                k.pop()
                k.append(")")
                dfs(l - 1, k)
                k.pop()
            

        res = list()
        dfs(0, [])
        return res 
```

[102. 二叉树的层序遍历](https://leetcode.cn/problems/binary-tree-level-order-traversal/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root: return []
        res, q = list(), deque([root,])
        while q:
            n = len(q)
            tmp = list()
            for _ in range(n):
                curr = q.popleft()
                tmp.append(curr.val)
                if curr.left: q.append(curr.left)
                if curr.right: q.append(curr.right)
            res.append(tmp)
        return res 
```

[199. 二叉树的右视图](https://leetcode.cn/problems/binary-tree-right-side-view/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root: return []
        res, q = list(), deque([root,])
        while q:
            n = len(q)
            k = 0
            for _ in range(n):
                curr = q.popleft()
                k = curr.val
                if curr.left: q.append(curr.left)
                if curr.right: q.append(curr.right)
            res.append(k)
        return res 
```

[92. 反转链表 II](https://leetcode.cn/problems/reverse-linked-list-ii/)

- 参考：https://leetcode.cn/problems/reverse-linked-list-ii/solution/fan-zhuan-lian-biao-ii-by-leetcode-solut-teyq/

```python
class Solution:
    def reverseBetween(self, head: ListNode, left: int, right: int) -> ListNode:
        def reverse_linked_list(head: ListNode):
            # 也可以使用递归反转一个链表
            pre = None
            cur = head
            while cur:
                next = cur.next # 1
                cur.next = pre # 2
                pre = cur # 3
                cur = next # 4 
                # 1、4 是往后走

        # 因为头节点有可能发生变化，使用虚拟头节点可以避免复杂的分类讨论
        dummy_node = ListNode(-1)
        dummy_node.next = head
        pre = dummy_node
        # 第 1 步：从虚拟头节点走 left - 1 步，来到 left 节点的前一个节点
        # 建议写在 for 循环里，语义清晰
        for _ in range(left - 1):
            pre = pre.next

        # 第 2 步：从 pre 再走 right - left + 1 步，来到 right 节点
        right_node = pre
        for _ in range(right - left + 1):
            right_node = right_node.next
        # 第 3 步：切断出一个子链表（截取链表）
        left_node = pre.next
        curr = right_node.next

        # 注意：切断链接
        pre.next = None
        right_node.next = None

        # 第 4 步：同第 206 题，反转链表的子区间
        reverse_linked_list(left_node)
        # 第 5 步：接回到原来的链表中
        pre.next = right_node
        left_node.next = curr
        return dummy_node.next

```




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
    - [200. 岛屿数量](https://leetcode.cn/problems/number-of-islands/)
    - [22. 括号生成](https://leetcode.cn/problems/generate-parentheses/)
    
- 排序
    - [56. 合并区间](https://leetcode.cn/problems/merge-intervals/)

- 二叉树
    - [102. 二叉树的层序遍历](https://leetcode.cn/problems/binary-tree-level-order-traversal/)
    - [199. 二叉树的右视图(层序遍历的小改动)](https://leetcode.cn/problems/binary-tree-right-side-view/)
