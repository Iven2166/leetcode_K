


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

[剑指 Offer 48. 最长不含重复字符的子字符串](https://leetcode-cn.com/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof/)

长度为 NN 的字符串共有 $\frac{(1+N)N}{2}$ 个子字符串（复杂度为 $O(N^2)$，判断长度为 NN 的字符串是否有重复字符的复杂度为 $O(N)$，因此本题使用暴力法解决的复杂度为 $O(N^3)$ .考虑使用动态规划降低时间复杂度。
[参考1](https://leetcode-cn.com/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof/solution/mian-shi-ti-48-zui-chang-bu-han-zhong-fu-zi-fu-d-9/)

- 时间复杂度 $O(N)$ ：遍历s
- 空间复杂度 $O(N)$ ：可以优化为 $O(1)$
```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if s == '':
            return 0
        curr_str = ''
        curr_dp, max_dp = 0, 0
        for i in s:
            # print(i, '-')
            if i not in curr_str:
                curr_str += i
                curr_dp += 1
            else:
                # print(curr_str, curr_str.find(i))
                curr_str = curr_str[curr_str.find(i)+1:] # get the first same str in the first place
                # print(curr_str)
                curr_str += i
                curr_dp = len(curr_str)
            # print(curr_str)
            max_dp = max(max_dp, curr_dp)
        return max_dp
```

## 第 11 天 双指针（简单）
[剑指 Offer 18. 删除链表的节点](https://leetcode-cn.com/problems/shan-chu-lian-biao-de-jie-dian-lcof/)

给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。返回删除后的链表的头节点。

- 复制一个链表的思路
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def deleteNode(self, head: ListNode, val: int) -> ListNode:
        if head.val == val:
            return head.next 
        pre = head 
        while head.next is not None:
            if head.next.val == val:
                head.next = head.next.next 
                break
            else:
                head = head.next 
        return pre
```

- 指针，指向目前（curr）以及前一个（pre）的思想 [参考1](https://leetcode-cn.com/problems/shan-chu-lian-biao-de-jie-dian-lcof/solution/mian-shi-ti-18-shan-chu-lian-biao-de-jie-dian-sh-2/)
```python
class Solution:
    def deleteNode(self, head: ListNode, val: int) -> ListNode:
        if head.val == val: return head.next
        pre, cur = head, head.next
        while cur and cur.val != val:
            pre, cur = cur, cur.next
        if cur: pre.next = cur.next
        return head
```

[剑指 Offer 22. 链表中倒数第k个节点](https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/)

输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。

例如，一个链表有 6 个节点，从头节点开始，它们的值依次是 1、2、3、4、5、6。这个链表的倒数第 3 个节点是值为 4 的节点。

- 两次遍历

执行用时：
40 ms
, 在所有 Python3 提交中击败了
37.71%
的用户
内存消耗：
14.9 MB
, 在所有 Python3 提交中击败了
73.97%
的用户
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
        curr = head 
        n = 0
        while curr:
            n += 1
            curr = curr.next 
        # print(n)
        curr = head
        i = 0
        while n - i > k:
            i += 1
            curr = curr.next 
        return curr
```

- 快慢指针法

执行用时：
32 ms
, 在所有 Python3 提交中击败了
86.62%
的用户
内存消耗：
14.9 MB
, 在所有 Python3 提交中击败了
54.37%
的用户
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
        fast, slow = head, head 
        while fast and k>0:
            fast, k = fast.next, k-1
        while slow and fast:
            fast, slow = fast.next, slow.next 
        return slow 
```

## 第 12 天 双指针（简单）

[剑指 Offer 25. 合并两个排序的链表](https://leetcode-cn.com/problems/he-bing-liang-ge-pai-xu-de-lian-biao-lcof/)

输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        ans = ListNode()
        curr = ans 
        while l1 and l2:
            if l1.val<=l2.val:
                curr.next = ListNode(l1.val)
                l1 = l1.next 
            else:
                curr.next = ListNode(l2.val)
                l2 = l2.next
            curr = curr.next 
        curr.next = l1 if l1 else l2 # 简洁写法
        return ans.next
```

[剑指 Offer 52. 两个链表的第一个公共节点](https://leetcode-cn.com/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/)

输入两个链表，找出它们的第一个公共节点。

[参考1](https://leetcode-cn.com/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/solution/liang-ge-lian-biao-de-di-yi-ge-gong-gong-pzbs/)

执行用时：
136 ms
, 在所有 Python3 提交中击败了
81.82%
的用户
内存消耗：
30 MB
, 在所有 Python3 提交中击败了
17.90%
的用户
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        # 本来想把headA的末尾接上headB，如果是循环则可以判断为有交集，但ab都不能有结构变化
        if not headA or not headB:
            return None 
        currA, currB = headA, headB
        while currA != currB:
            currA = headB if not currA else currA.next
            currB = headA if not currB else currB.next
        return currA if currA else None 
```

## 第 13 天 双指针（简单）
[剑指 Offer 21. 调整数组顺序使奇数位于偶数前面](https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/)

输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数在数组的前半部分，所有偶数在数组的后半部分。

- 自己解法

执行用时：
56 ms
, 在所有 Python3 提交中击败了
26.25%
的用户
内存消耗：
19.1 MB
, 在所有 Python3 提交中击败了
26.89%
的用户
```python
class Solution:
    def exchange(self, nums: List[int]) -> List[int]:
        n = len(nums)
        left, right = 0, n-1
        while left<=right:
            if nums[left]%2==0 and nums[right]%2==1:
                nums[left], nums[right] = nums[right], nums[left]
            elif nums[left]%2==0 and nums[right]%2==0:
                right -= 1
            elif nums[left]%2==1 and nums[right]%2==1:
                left += 1
            else:
                left, right = left + 1, right - 1
        return nums
```

- [简洁解法](https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/solution/mian-shi-ti-21-diao-zheng-shu-zu-shun-xu-shi-qi-4/)
：用位运算提升速度, 指针移动条件其实可以合并
  
执行用时：
44 ms
, 在所有 Python3 提交中击败了
80.96%
的用户
内存消耗：
19 MB
, 在所有 Python3 提交中击败了
62.57%
的用户
```python
class Solution:
    def exchange(self, nums: List[int]) -> List[int]:
        i, j = 0, len(nums) - 1
        while i < j:
            while i < j and nums[i] & 1 == 1: i += 1
            while i < j and nums[j] & 1 == 0: j -= 1
            nums[i], nums[j] = nums[j], nums[i]
        return nums

# 作者：jyd
# 链接：https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/solution/mian-shi-ti-21-diao-zheng-shu-zu-shun-xu-shi-qi-4/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

[剑指 Offer 57. 和为s的两个数字](https://leetcode-cn.com/problems/he-wei-sde-liang-ge-shu-zi-lcof/)

输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。

关键条件：递增排序

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        left, right = 0, len(nums) - 1
        while left < right:
            while nums[left] + nums[right] > target: right -= 1
            while nums[left] + nums[right] < target: left += 1
            if nums[left]+nums[right]==target: return [nums[left], nums[right]]
        return None
```

[剑指 Offer 58 - I. 翻转单词顺序](https://leetcode-cn.com/problems/fan-zhuan-dan-ci-shun-xu-lcof/)

输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。为简单起见，标点符号和普通字母一样处理。例如输入字符串"I am a student. "，则输出"student. a am I"。

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        s = s.strip()
        i = j = len(s) - 1 
        res = []
        # search from the last string 
        while i >= 0:
            while i >= 0 and s[i]!=' ': i -= 1
            res.append(s[i + 1 : j + 1].replace(' ',''))
            while s[i]==' ': i -= 1
            j = i
        return ' '.join(res)
```

## 第 14 天 搜索与回溯算法（中等）

[剑指 Offer 12. 矩阵中的路径](https://leetcode-cn.com/problems/ju-zhen-zhong-de-lu-jing-lcof/)

题目：给定一个 $m x n$ 二维字符网格 $board$ 和一个字符串单词 $word$ 。如果 $word$ 存在于网格中，返回 $true$ ；否则，返回 $false$ 。
单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。


- [答案参考1](https://leetcode-cn.com/problems/ju-zhen-zhong-de-lu-jing-lcof/solution/mian-shi-ti-12-ju-zhen-zhong-de-lu-jing-shen-du-yo/)

解题思路：
本问题是典型的矩阵搜索问题，可使用 深度优先搜索（DFS）+ 剪枝 解决。

深度优先搜索： 可以理解为暴力法遍历矩阵中所有字符串可能性。DFS 通过递归，先朝一个方向搜到底，再回溯至上个节点，沿另一个方向搜索，以此类推。

*这个一个理解DFS的很好案例。思路：*

1. 创建helper函数，进行DFS
2. 判断三种访问失败的条件：（1）超过边界（2）当前的字符并不是预估的 word[k] （走了k步）
3. 判断成功条件：k步完成
4. 下一层判断：先把本步已访问过记录下来，再用res作为下一层的判断，回溯后，恢复本步的字符

```python3
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        def dfs(i, j, k):
            if not 0 <= i < len(board) or not 0 <= j < len(board[0]) or board[i][j] != word[k]:
                return False
            if k == len(word) - 1:
                return True
            board[i][j] = '' # 当作当前字符已经访问过，所以下一层没有办法走回头路
            res = dfs(i-1, j, k+1) or dfs(i+1, j, k+1) or dfs(i, j-1, k+1) or dfs(i, j+1, k+1)
            board[i][j] = word[k] # 1. 回溯时需要恢复[i][j]位置的字符 2. 为何用word[k]?-由于是匹配上了，所以可用word[k]
            return res 
        
        for i in range(len(board)):
            for j in range(len(board[0])):
                if dfs(i, j, 0):
                    return True 
        return False
```

[剑指 Offer 13. 机器人的运动范围](https://leetcode-cn.com/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof/)

题目：地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？

[参考1](https://leetcode-cn.com/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof/solution/ji-qi-ren-de-yun-dong-fan-wei-by-leetcode-solution/)
- 解法1：从左上往右下搜索
```python3
class Solution:
    def movingCount(self, m: int, n: int, k: int) -> int:
        s1 = set([(0,0), ])
        for i in range(m):
            for j in range(n):
                if ((i-1, j) in s1 or (i, j-1) in s1) and i//10 + i%10 + j//10 + j%10 <= k:
                    s1.add((i, j))
        # print(s1)
        return len(s1)
```

## 第 15 天 搜索与回溯算法（中等）

[剑指 Offer 34. 二叉树中和为某一值的路径](https://leetcode-cn.com/problems/er-cha-shu-zhong-he-wei-mou-yi-zhi-de-lu-jing-lcof/)

给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。

叶子节点 是指没有子节点的节点。


```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: TreeNode, target: int) -> List[List[int]]:
        # 到了叶子结点，还没到target，返回false
        total_ans = []
        ans = []
        def dfs(curr, k):
            if not curr:
                return 
            ans.append(curr.val)
            if not curr.left and not curr.right and k + curr.val == target:
                total_ans.append(list(ans)) # 注意这里不能写ans，因为是引用，而ans最终是回撤清空的；不做浅层copy
            dfs(curr.left, k + curr.val) 
            dfs(curr.right, k + curr.val)
            ans.pop()

        dfs(root, 0)
        return total_ans
```

- DFS写法
```python
ans, total_ans = [], []
def dfs(root):
    if not root:
        return 
    ans.append(root.val)
    if not curr.left and not curr.right:
        total_ans.append(list(ans))
    dfs(root.left)
    dfs(root.right)
    ans.pop()
```

[剑指 Offer 36. 二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)

输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的循环双向链表。要求不能创建任何新的节点，只能调整树中节点指针的指向。

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
"""
class Solution:
    def treeToDoublyList(self, root: 'Node') -> 'Node':
        if not root:
            return None
        stack = list()
        def dfs(curr):
            if not curr:
                return 
            if curr.left:
                dfs(curr.left)
            stack.append(curr)
            if curr.right:
                dfs(curr.right)
        dfs(root)

        n = len(stack)
        for i, j in enumerate(stack):
            j.left = stack[(i + n - 1)%n]
            j.right = stack[(i + n + 1)%n]
        return stack[0]
```

[剑指 Offer 54. 二叉搜索树的第k大节点](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)

给定一棵二叉搜索树，请找出其中第 $k$ 大的节点的值。

- 存储后进行判断

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def kthLargest(self, root: TreeNode, k: int) -> int:
        if not root:
            return None
        stack = []
        def dfs(curr):
            if not curr:
                return 
            if curr.left:
                dfs(curr.left)
            stack.append(curr.val)
            if curr.right:
                dfs(curr.right)
        
        dfs(root)
        return stack[-k]
```

- 提前判断退出 [参考题解1](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/solution/mian-shi-ti-54-er-cha-sou-suo-shu-de-di-k-da-jie-d/)

复杂度分析：
时间复杂度 O(N)： 当树退化为链表时（全部为右子节点），无论 k 的值大小，递归深度都为 N

空间复杂度 O(N)： 当树退化为链表时（全部为右子节点），系统使用 O(N) 大小的栈空间。


```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def kthLargest(self, root: TreeNode, k: int) -> int:
        def dfs(curr):
            if not curr:
                return 
            dfs(curr.right)
            if self.k == 0:
                return 
            self.k -= 1
            if self.k == 0:
                self.res = curr.val 
            dfs(curr.left)
        
        self.k = k 
        dfs(root)
        return self.res 
```

## 第 16 天 排序（简单）

[剑指 Offer 45. 把数组排成最小的数](https://leetcode-cn.com/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/)

输入一个非负整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。


- [参考](https://leetcode-cn.com/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/solution/mian-shi-ti-45-ba-shu-zu-pai-cheng-zui-xiao-de-s-4/)

用 sort(key = functools.cmp_to_key(func))

```python
class Solution:
    def minNumber(self, nums: List[int]) -> str:
        def sort_value(a,b):
            x, y = a + b, b + a
            if x > y: return 1
            elif x < y: return -1  # 相当于a less than b，a应该排在前面
            else: return 0

        strs = [str(num) for num in nums]
        strs.sort(key = functools.cmp_to_key(sort_value))
        return ''.join(strs)
```

[剑指 Offer 61. 扑克牌中的顺子](https://leetcode-cn.com/problems/bu-ke-pai-zhong-de-shun-zi-lcof/)

从若干副扑克牌中随机抽 5 张牌，判断是不是一个顺子，即这5张牌是不是连续的。2～10为数字本身，A为1，J为11，Q为12，K为13，而大、小王为 0 ，可以看成任意数字。A 不能视为 14。

```python
class Solution:
    def isStraight(self, nums: List[int]) -> bool:
        s = set()
        max_num, min_num = 0, 14
        for num in nums:
            if num == 0: continue
            max_num = max(max_num, num)
            min_num = min(min_num, num)
            if num in s:
                return False
            s.add(num)
        if max_num - min_num < 5:
            return True 
        else:
            return False
```

## 第 17 天 排序（中等）

[剑指 Offer 40. 最小的k个数](https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/)

输入整数数组 arr ，找出其中最小的 k 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。


- 快排( [图解](https://blog.csdn.net/Adusts/article/details/80882649?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165159282216782395360412%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=165159282216782395360412&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-80882649.142^v9^pc_search_result_control_group,157^v4^control&utm_term=%E5%BF%AB%E6%8E%92&spm=1018.2226.3001.4187) )

```python
def quick_sort(arr, l, r): # 快排
    if l >= r:
        return 
    i, j = l, r
    while i < j:
        while i < j and arr[j] >= arr[l]: j -= 1
        while i < j and arr[i] <= arr[l]: i += 1
        arr[i], arr[j] = arr[i], arr[j]
    arr[l], arr[i] = arr[i], arr[l]
    quick_sort(arr, l, i - 1)
    quick_sort(arr, i + 1, r)
```

[剑指 Offer 41. 数据流中的中位数(困难)](https://leetcode-cn.com/problems/shu-ju-liu-zhong-de-zhong-wei-shu-lcof/)

如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值 *排序之后* 位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。

- [参考](https://leetcode-cn.com/problems/shu-ju-liu-zhong-de-zhong-wei-shu-lcof/solution/mian-shi-ti-41-shu-ju-liu-zhong-de-zhong-wei-shu-y/)
  
  - 核心逻辑：实时地分割出来大的一半和小的一半，"分割线"为中位数：所以从大的部分里面找最小的，从小的部分找最大的，一取平均就是中位数。
  - 实现
    - 比中位数大的部分 & 能够返回最小的一个：小顶堆
    - 比中位数小的部分 & 能够返回最大的一个：大顶堆（由于Python 中 heapq 模块是小顶堆。所以实现 大顶堆 方法： 小顶堆的插入和弹出操作均将元素 取反 即可。）
  

![img_1.png](img_1.png)


```python
from heapq import * 

class MedianFinder:
    def __init__(self):
        self.A = [] # 小顶堆，保存较大的一半
        self.B = [] # 大顶堆，保存较小的一半

    def addNum(self, num: int) -> None:
        if len(self.A)!=len(self.B):
            # heappush(self.A, num)
            # heappush(self.B, -heappop(self.A))
            heappush(self.B, -heappushpop(self.A, num)) # 代替两行更加简洁
        else:
            # heappush(self.B, -num)
            # heappush(self.A, -heappop(self.B))
            heappush(self.A, -heappushpop(self.B, -num)) # 更加简洁
    
    def findMedian(self) -> float:
        return self.A[0] if len(self.A)!=len(self.B) else (self.A[0] - self.B[0]) / 2.0
```

## 第 18 天 搜索与回溯算法（中等）

[剑指 Offer 55 - I. 二叉树的深度](https://leetcode-cn.com/problems/er-cha-shu-de-shen-du-lcof/)

输入一棵二叉树的根节点，求该树的深度。从根节点到叶节点依次经过的节点（含根、叶节点）形成树的一条路径，最长路径的长度为树的深度。

- 普通的递归 recursion

```python
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root: return 0 
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
```

- DFS

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        self.depth, self.depth_max = 0, 0
        def dfs(curr):
            # print(curr.val, self.depth)
            if not curr:
                return 
            self.depth += 1
            if not curr.left and not curr.right:
                self.depth_max = max(self.depth_max, self.depth)
                return 
            dfs(curr.left)
            if curr.left:
                self.depth -= 1
            dfs(curr.right)
            if curr.right:
                self.depth -= 1
            
        dfs(root)
        return self.depth_max
```

- 层序遍历（BFS）:在时空上应该是最快的

```python
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root: return 0 
        queue, res = [root, ], 0
        while queue:
            tmp = []
            for node in queue:
                if node.left: tmp.append(node.left)
                if node.right: tmp.append(node.right)
            queue = tmp 
            res += 1
        return res
```

[剑指 Offer 55 - II. 平衡二叉树](https://leetcode-cn.com/problems/ping-heng-er-cha-shu-lcof/)

输入一棵二叉树的根节点，判断该树是不是平衡二叉树。如果某二叉树中任意节点的左右子树的深度相差不超过1，那么它就是一棵平衡二叉树。

- 时间复杂度：$O(n^2)$，其中 $n$ 是二叉树中的节点个数。
最坏情况下，二叉树是满二叉树，需要遍历二叉树中的所有节点。
对于节点 $p$，如果它的高度是 $d$，则 $cnt(p)$ 最多会被调用 $d$ 次
  （即遍历到它的每一个祖先节点时）。对于平均的情况，一棵树的高度 $h$ 满足 $O(h)=O(log n)$，因为 $d \leq h$，所以总时间复杂度为 $O(n log n)$。
  对于最坏的情况，二叉树形成链式结构，高度为 $O(n)$，此时总时间复杂度为 $O(n^2)$


```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        def cnt(curr):
            if not curr: 
                return 0 
            else:
                return max(cnt(curr.left), cnt(curr.right)) + 1

        if not root:
            return True
        if abs(cnt(root.left) - cnt(root.right)) >= 2:
            return False
        else:
            return self.isBalanced(root.left) and self.isBalanced(root.right)
```

## 第 19 天 搜索与回溯算法（中等）

[剑指 Offer 64. 求1+2+…+n](https://leetcode-cn.com/problems/qiu-12n-lcof/)

求 1+2+...+n ，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。



[剑指 Offer 68 - I. 二叉搜索树的最近公共祖先](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-zui-jin-gong-gong-zu-xian-lcof/)

- 没有充分利用二叉搜索树性质的DFS：时间复杂度排名约10%

先找两个节点的共同路径，然后再逆序寻找第一个出现的共同节点（足够深）

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        res = []
        self.ans = []
        def dfs(curr, k):
            if curr:
                res.append(curr)
                if curr == k:
                    self.ans = list(res)
                dfs(curr.left, k)
                dfs(curr.right, k)
                res.pop()
            
        dfs(root, p)
        ans1 = self.ans
        self.ans = []
        dfs(root, q)
        ans2 = self.ans
        # print(ans1, ans2)
        for i in ans1[::-1]:
            for j in ans2[::-1]:
                if i == j:
                    return i
```

- 考虑二叉搜索树性质的，在找路径时，需要判断根节点和需要寻找的节点大小关系：如果要找的节点小，那么在左子树，反之则在右子树

时间复杂度：
```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        def findPath(curr, k):
            path = list()
            while curr.val != k.val:
                path.append(curr)
                if k.val < curr.val:
                    curr = curr.left 
                else:
                    curr = curr.right
            path.append(k)
            return path 
        
        path1, path2 = findPath(root, p), findPath(root, q)
        for i in path1[::-1]:
            for j in path2[::-1]:
                if i == j:
                    return i
```

- [参考：一次遍历](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-zui-jin-gong-gong-zu-xian-lcof/solution/er-cha-sou-suo-shu-de-zui-jin-gong-gong-0wpw1/)

把两个节点放在同一个判断力，一次遍历。如果都小，则都在左边；如果都大，那么都在右边；其他情况说明是分叉口。

```python
class Solution:
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        ancestor = root
        while True:
            if p.val < ancestor.val and q.val < ancestor.val:
                ancestor = ancestor.left
            elif p.val > ancestor.val and q.val > ancestor.val:
                ancestor = ancestor.right
            else:
                break
        return ancestor
```

## 第 20 天 分治算法（中等）

[剑指 Offer 07. 重建二叉树](https://leetcode-cn.com/problems/zhong-jian-er-cha-shu-lcof/)

输入某二叉树的前序遍历和中序遍历的结果，请构建该二叉树并返回其根节点。

- 逻辑：其实简单题，因为前序遍历（根，左，右）和中序遍历（左，根，右）是能够切分开来，问题即定位根节点
- 复杂度：如果用列表的index方法，那么定位根节点需要$O(N)$

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
         
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if not preorder and not inorder:
            return None
        if len(preorder)==len(inorder)==1:
            return TreeNode(preorder[0])
        mid = inorder.index(preorder[0])
        curr = TreeNode(preorder[0])
        curr.left = self.buildTree(
          preorder[1: mid + 1], inorder[0: mid]
        )
        curr.right = self.buildTree(
          preorder[mid + 1:], inorder[ mid + 1:]
        )
        return curr   
```

- 优化：用哈希表来优化查询速度(从 17% ——> 67%)

```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        def helper(pre_pos, in_pos):
            pre_left, pre_right = pre_pos
            in_left, in_right = in_pos
            if pre_left > pre_right:
                return None
            root = preorder[pre_left]
            mid = inorder_index[root]
            curr = TreeNode(root)
            inorder_gap = mid - in_left
            curr.left = helper(
              (pre_left + 1, pre_left + inorder_gap), (in_left, mid - 1)
            )
            curr.right = helper(
                (pre_left + inorder_gap + 1, pre_right), (mid + 1, in_right)
            )
            return curr 
        
        inorder_index = {element: i for i, element in enumerate(inorder)}
        n = len(inorder)
        return helper((0, n-1), (0, n-1))
```


[剑指 Offer 16. 数值的整数次方](https://leetcode-cn.com/problems/shu-zhi-de-zheng-shu-ci-fang-lcof/)

实现 pow(x, n) ，即计算 x 的 n 次幂函数。不得使用库函数，同时不需要考虑大数问题。

[解答：快速幂](https://leetcode-cn.com/problems/shu-zhi-de-zheng-shu-ci-fang-lcof/solution/mian-shi-ti-16-shu-zhi-de-zheng-shu-ci-fang-kuai-s/)

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if x == 0: # 因为下方有 1/x
            return x
        if n < 0:
            x, n = 1/x, - n
        res = 1
        while n:
            if n & 1:
                res *= x
            x *= x 
            n >>= 1
        return res
```


[剑指 Offer 33. 二叉搜索树的后序遍历序列](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/)

输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回 true，否则返回 false。假设输入的数组的任意两个数字都互不相同。

```python
class Solution:
    def verifyPostorder(self, postorder: List[int]) -> bool:
        def helper(i, j):
            if i >= j:
                return True
            pos = i
            while postorder[pos] < postorder[j]:
                pos += 1
            mark = pos 
            while postorder[pos] > postorder[j]:
                pos += 1
            return pos == j and helper(i, mark - 1) and helper(mark, j - 1)
        return helper(0, len(postorder) - 1)
```

## 第 21 天 位运算（简单）

[剑指 Offer 15. 二进制中1的个数](https://leetcode-cn.com/problems/er-jin-zhi-zhong-1de-ge-shu-lcof/)

编写一个函数，输入是一个无符号整数（以二进制串的形式），返回其二进制表达式中数字位数为 '1' 的个数（也被称为 汉明重量).）。

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        cnt = 0
        while n > 0:
            cnt += n & 1
            n >>= 1
        return cnt 
```


[剑指 Offer 65. 不用加减乘除做加法](https://leetcode-cn.com/problems/bu-yong-jia-jian-cheng-chu-zuo-jia-fa-lcof/)

写一个函数，求两个整数之和，要求在函数体内不得使用 “+”、“-”、“*”、“/” 四则运算符号。

[参考](https://leetcode-cn.com/problems/bu-yong-jia-jian-cheng-chu-zuo-jia-fa-lcof/solution/mian-shi-ti-65-bu-yong-jia-jian-cheng-chu-zuo-ji-7/)

```python
class Solution:
    def add(self, a: int, b: int) -> int:
        x = 0xffffffff
        a, b = a & x, b & x
        while b != 0:
            a, b = (a ^ b), (a & b) << 1 & x
        return a if a <= 0x7fffffff else ~(a ^ x)
```

## 第 23 天 数学（简单）

[剑指 Offer 39. 数组中出现次数超过一半的数字](https://leetcode-cn.com/problems/shu-zu-zhong-chu-xian-ci-shu-chao-guo-yi-ban-de-shu-zi-lcof/)

数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。

- 解法
  - 哈希表
  - 排序后找中间位置数字
  - 摩尔投票法（符合空间复杂度 $O(1)$ ）[link](https://leetcode-cn.com/problems/shu-zu-zhong-chu-xian-ci-shu-chao-guo-yi-ban-de-shu-zi-lcof/solution/mian-shi-ti-39-shu-zu-zhong-chu-xian-ci-shu-chao-3/)

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        votes = 0
        for num in nums:
            if votes == 0:
                x = num 
            votes += 1 if num == x else -1 
        return x
```

[剑指 Offer 66. 构建乘积数组](https://leetcode-cn.com/problems/gou-jian-cheng-ji-shu-zu-lcof/)

给定一个数组 A[0,1,…,n-1]，请构建一个数组 B[0,1,…,n-1]，其中 B[i] 的值是数组 A 中除了下标 i 以外的元素的积, 即 B[i]=A[0]×A[1]×…×A[i-1]×A[i+1]×…×A[n-1]。不能使用除法。


- 循规蹈矩的解法，判断每个位置的两端的乘积

```python
class Solution:
    def constructArr(self, a: List[int]) -> List[int]:
        c, d = list(a), list(a) 
        for i in range(1, len(c)):
            c[i] = c[i-1] * c[i]
        for i in range(len(d)-2, -1, -1):
            d[i] = d[i+1] * d[i]
        ans = [0 for _ in range(len(a))]
        for i in range(len(ans)):
            if i == 0:
                ans[i] = d[1]
            elif i == len(ans)-1:
                ans[i] = c[len(ans)-2]
            else:
                ans[i] = c[i-1] * d[i+1]   
        return ans 
```
- 优化空间复杂度: [图解：上下三角](https://leetcode-cn.com/problems/gou-jian-cheng-ji-shu-zu-lcof/solution/mian-shi-ti-66-gou-jian-cheng-ji-shu-zu-biao-ge-fe/)

```python
class Solution:
    def constructArr(self, a: List[int]) -> List[int]:
        b, tmp = [1] * len(a), 1
        for i in range(1, len(a)):
            b[i] = b[i-1] * a[i-1]
        for i in range(len(a) - 2, -1, -1):
            tmp *= a[i + 1]
            b[i] *= tmp 
        return b 
```

## 第 24 天 数学（中等）

[剑指 Offer 14- I. 剪绳子](https://leetcode-cn.com/problems/jian-sheng-zi-lcof/)

给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m-1] 。请问 k[0]*k[1]*...*k[m-1] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

```python
class Solution:
    def cuttingRope(self, n: int) -> int:
        if n <= 3: 
            return n - 1
        a, b = n // 3, n % 3
        if b == 0:
            return int(math.pow(3, a))
        elif b == 1:
            return int(math.pow(3, a - 1) * 4)
        else:
            return int(math.pow(3, a) * 2)
```

- 解答：
  - 注：Python 中常见有三种幂计算函数： * 和 pow() 的时间复杂度均为 O(\log a)；而 math.pow() 始终调用 C 库的 pow() 函数，其执行浮点取幂，时间复杂度为 O(1)。
  - 应该尽可能地瓜分为长度为3的分段[数学推导](https://leetcode-cn.com/problems/jian-sheng-zi-lcof/solution/mian-shi-ti-14-i-jian-sheng-zi-tan-xin-si-xiang-by/)

[剑指 Offer 14- II. 剪绳子 II](https://leetcode-cn.com/problems/jian-sheng-zi-ii-lcof/)

- [解法参考](https://leetcode-cn.com/problems/jian-sheng-zi-ii-lcof/solution/mian-shi-ti-14-ii-jian-sheng-zi-iitan-xin-er-fen-f/)
  - 循环求余
  - 快速幂求余
  
  
[剑指 Offer 57 - II. 和为s的连续正数序列](https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/)

输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。

序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。

- 利用求和公式
    - 举一反三：如果不是连续正整数序列，而是间隔2如何处理？——实际上就是修改求和公式，再更新公式
  
```python
class Solution:
    def findContinuousSequence(self, target: int):
        i, j, res = 1, 2, []
        while i < j:
            j = (-1 + (1 + 4 * (2 * target + i * i - i)) ** 0.5) / 2
            if i < j and j == int(j):
                res.append(list(range(i, int(j) + 1)))
            i += 1
        return res
```


[剑指 Offer 62. 圆圈中最后剩下的数字](https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/)

- [约瑟夫环：加强理解](https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/solution/jian-zhi-offer-62-yuan-quan-zhong-zui-ho-dcow/)



## 第 25 天 模拟（中等）

[剑指 Offer 29. 顺时针打印矩阵](https://leetcode-cn.com/problems/shun-shi-zhen-da-yin-ju-zhen-lcof/)

输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        res = []
        while matrix:
            res += matrix.pop(0)
            matrix = list(zip(*matrix))[::-1]
        return res
```


- [设定边界，并且边界变化](https://leetcode-cn.com/problems/shun-shi-zhen-da-yin-ju-zhen-lcof/solution/mian-shi-ti-29-shun-shi-zhen-da-yin-ju-zhen-she-di/)

```python
class Solution:
    def spiralOrder(self, matrix:[[int]]) -> [int]:
        if not matrix: return []
        l, r, t, b, res = 0, len(matrix[0]) - 1, 0, len(matrix) - 1, []
        while True:
            for i in range(l, r + 1): res.append(matrix[t][i]) # left to right
            t += 1
            if t > b: break
            for i in range(t, b + 1): res.append(matrix[i][r]) # top to bottom
            r -= 1
            if l > r: break
            for i in range(r, l - 1, -1): res.append(matrix[b][i]) # right to left
            b -= 1
            if t > b: break
            for i in range(b, t - 1, -1): res.append(matrix[i][l]) # bottom to top
            l += 1
            if l > r: break
        return res
```

[剑指 Offer 31. 栈的压入、弹出序列](https://leetcode-cn.com/problems/zhan-de-ya-ru-dan-chu-xu-lie-lcof/)

输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如，序列 {1,2,3,4,5} 是某栈的压栈序列，序列 {4,5,3,2,1} 是该压栈序列对应的一个弹出序列，但 {4,3,5,1,2} 就不可能是该压栈序列的弹出序列。

```python 
class Solution:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        stack, i = [], 0
        for num in pushed:
            stack.append(num)
            while stack and stack[-1] == popped[i]:
                stack.pop()
                i += 1
        return not stack
```

## 第 26 天 字符串（中等）

[剑指 Offer 20. 表示数值的字符串](https://leetcode-cn.com/problems/biao-shi-shu-zhi-de-zi-fu-chuan-lcof/)


[剑指 Offer 67. 把字符串转换成整数](https://leetcode-cn.com/problems/ba-zi-fu-chuan-zhuan-huan-cheng-zheng-shu-lcof/)


## 第 27 天 栈与队列（困难）

[剑指 Offer 59 - I. 滑动窗口的最大值](https://leetcode-cn.com/problems/hua-dong-chuang-kou-de-zui-da-zhi-lcof/)

```python
from heapq import *
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if not nums:
            return []
        # 维护一个小顶堆，存储当前值以及位置，记录当前的窗口范围，结合判断值是否在窗口里以及是否最大。
        ans = []
        curr = []
        i = 0
        n = len(nums)
        while i + k - 1 <= n - 1:
            if i == 0: # 初始化
                for pos, j in enumerate(nums[0:i+k]):
                    heappush(curr, (-j, pos))
            else:
                heappush(curr, (- nums[i + k - 1], i + k - 1))
            while True:
                val, pos = curr[0]
                val *= -1
                if pos >= i and pos <= i + k - 1:
                    ans.append(val)
                    break
                else:
                    heappop(curr)
            i += 1
        return ans
```

[剑指 Offer 59 - II. 队列的最大值](https://leetcode-cn.com/problems/dui-lie-de-zui-da-zhi-lcof/)

- 参考题解[双向队列，重点在于递减序列的维护](https://leetcode-cn.com/problems/dui-lie-de-zui-da-zhi-lcof/solution/jian-zhi-offer-59-ii-dui-lie-de-zui-da-z-0pap/)

```python
import queue
class MaxQueue:

    def __init__(self):
        self.q = queue.Queue()
        self.stack_max = queue.deque()

    def max_value(self) -> int:
        return self.stack_max[0] if self.stack_max else -1

    def push_back(self, value: int) -> None:
        self.q.put(value)
        while self.stack_max and self.stack_max[-1] < value:
            self.stack_max.pop()
        self.stack_max.append(value)
        return None

    def pop_front(self) -> int:
        if self.q.empty(): 
            return -1
        p = self.q.get()
        if self.stack_max[0] == p:
            self.stack_max.popleft()
        return p


# Your MaxQueue object will be instantiated and called as such:
# obj = MaxQueue()
# param_1 = obj.max_value()
# obj.push_back(value)
# param_3 = obj.pop_front()
```

## 第 28 天 搜索与回溯算法（困难）

[剑指 Offer 37. 序列化二叉树](https://leetcode-cn.com/problems/xu-lie-hua-er-cha-shu-lcof/)

请实现两个函数，分别用来序列化和反序列化二叉树。

你需要设计一个算法来实现二叉树的序列化与反序列化。这里不限定你的序列 / 反序列化算法执行逻辑，你只需要保证一个二叉树可以被序列化为一个字符串并且将这个字符串反序列化为原始的树结构。

```python
import collections
class Codec:
    def serialize(self, root):
        if not root: return "[]"
        res = []
        queue = collections.deque()
        queue.append(root)
        while queue:
            curr = queue.popleft()
            if curr:
                res.append(str(curr.val))
                queue.append(curr.left)
                queue.append(curr.right)
            else:
                res.append("null")  
        return '[' + ','.join(res) + ']'
    
    def deserialize(self, data):
        if data=="[]": return 
        vals = data[1:-1].split(',')
        i = 1
        queue = collections.deque()
        root = TreeNode(int(vals[0]))
        queue.append(root)
        while queue:
            curr = queue.popleft()
            if vals[i]!="null":
                curr.left = TreeNode(int(vals[i]))
                queue.append(curr.left)
            i += 1
            if vals[i]!="null":
                curr.right = TreeNode(int(vals[i]))
                queue.append(curr.right)
            i += 1
        return root
```


[剑指 Offer 38. 字符串的排列](https://leetcode-cn.com/problems/zi-fu-chuan-de-pai-lie-lcof/)

- [参考](https://leetcode-cn.com/problems/zi-fu-chuan-de-pai-lie-lcof/solution/mian-shi-ti-38-zi-fu-chuan-de-pai-lie-hui-su-fa-by/)

```python
class Solution:
    def permutation(self, s: str) -> List[str]:
        c, res = list(s), []
        def dfs(x):
            if x == len(c) - 1:
                res.append(''.join(c))   # 添加排列方案
                return
            dic = set()
            for i in range(x, len(c)):
                if c[i] in dic: 
                    continue # 重复，因此剪枝
                dic.add(c[i])
                c[i], c[x] = c[x], c[i]  # 交换，将 c[i] 固定在第 x 位
                dfs(x + 1)               # 开启固定第 x + 1 位字符
                c[i], c[x] = c[x], c[i]  # 恢复交换
        dfs(0)
        return res
```

## 第 29 天 动态规划（困难）

[剑指 Offer 19. 正则表达式匹配](https://leetcode-cn.com/problems/zheng-ze-biao-da-shi-pi-pei-lcof/)

请实现一个函数用来匹配包含'. '和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（含0次）。在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但与"aa.a"和"ab*a"均不匹配。


- [参考](https://leetcode-cn.com/problems/zheng-ze-biao-da-shi-pi-pei-lcof/solution/zhu-xing-xiang-xi-jiang-jie-you-qian-ru-shen-by-je/)

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s) + 1, len(p) + 1
        dp = [[False] * n for _ in range(m)]
        dp[0][0] = True 
        for j in range(2, n, 2):
            dp[0][j] = dp[0][j - 2] and p[j - 1] == '*'
        for i in range(1, m):
            for j in range(1, n):
                if p[j - 1] == '*':
                    if dp[i][j - 2]: dp[i][j] = True 
                    if dp[i - 1][j] and s[i-1] == p[j-2]: dp[i][j] = True
                    if dp[i-1][j] and p[j-2] == '.': dp[i][j] = True
                else:
                    if dp[i - 1][j - 1] and s[i-1] == p[j-1]: dp[i][j] = True
                    if dp[i - 1][j - 1] and p[j-1] == '.': dp[i][j] = True
        return dp[-1][-1]
```

[剑指 Offer 49. 丑数](https://leetcode-cn.com/problems/chou-shu-lcof/)

我们把只包含质因子 2、3 和 5 的数称作丑数（Ugly Number）。求按从小到大的顺序的第 n 个丑数。

- 小顶堆，暴力解法，时间复杂度高（因为往后进行了多余的计算)
```python

```

- [维护三个指针，分别对应*2，*3，*5的最小值，避免往后多余的计算](https://leetcode-cn.com/problems/chou-shu-lcof/solution/mian-shi-ti-49-chou-shu-dong-tai-gui-hua-qing-xi-t/)
```python
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        dp, a, b, c = [1] * n, 0, 0, 0
        for i in range(1, n):
            n2, n3, n5 = dp[a]*2, dp[b]*3, dp[c]*5
            dp[i] = min(n2, n3, n5)
            if dp[i] == n2: a += 1
            if dp[i] == n3: b += 1
            if dp[i] == n5: c += 1
        return dp[-1]
```


[剑指 Offer 60. n个骰子的点数](https://leetcode-cn.com/problems/nge-tou-zi-de-dian-shu-lcof/)

把n个骰子扔在地上，所有骰子朝上一面的点数之和为s。输入n，打印出s的所有可能的值出现的概率。

```python
class Solution:
    def dicesProbability(self, n: int) -> List[float]:
        dp = [list() for _ in range(n)]
        for i in range(6):
            dp[0].append(1)
        if n == 1:
            return [i/6 for i in dp[0]]
        for i in range(1, n):
            dp[i] = [0] * 6 * (i + 1)
            dp[i][i] = 1
            j = i + 1
            while j < len(dp[i]):
                for m in range(max(j-6,0), min(j,len(dp[i-1]))):
                    dp[i][j] += dp[i-1][m]
                j += 1
        dp_i = dp[-1]
        dp_sum = sum(dp[-1])
        res = []
        for i in dp_i:
            if i != 0:
                res.append(i/dp_sum)
        return res
```

## 第 30 天 分治算法（困难）

[剑指 Offer 17. 打印从1到最大的n位数](https://leetcode-cn.com/problems/da-yin-cong-1dao-zui-da-de-nwei-shu-lcof/)

输入数字 n，按顺序打印出从 1 到最大的 n 位十进制数。比如输入 3，则打印出 1、2、3 一直到最大的 3 位数 999。

```python
class Solution:
    def printNumbers(self, n: int) -> List[int]:
        return [i for i in range(1,10**n)]
```

- [参考-dfs](https://leetcode-cn.com/problems/da-yin-cong-1dao-zui-da-de-nwei-shu-lcof/solution/mian-shi-ti-17-da-yin-cong-1-dao-zui-da-de-n-wei-2/)
```python
class Solution:
    def printNumbers(self, n: int) -> [int]:
        def dfs(x):
            if x == n: # 终止条件：已固定完所有位
                res.append(''.join(num)) # 拼接 num 并添加至 res 尾部
                return
            for i in range(10): # 遍历 0 - 9
                num[x] = str(i) # 固定第 x 位为 i
                dfs(x + 1) # 开启固定第 x + 1 位
        
        num = ['0'] * n # 起始数字定义为 n 个 0 组成的字符列表
        res = [] # 数字字符串列表
        dfs(0) # 开启全排列递归
        return ','.join(res)  # 拼接所有数字字符串，使用逗号隔开，并返回
```

[剑指 Offer 51. 数组中的逆序对](https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)

在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。

- 连结知识点：归并排序，在排序的同时，也会累积逆序的比较
- 时间复杂度 $O(N \log N)$： 其中 $N$ 为数组长度；归并排序使用 $O(N \log N)$ 时间
- [参考](https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/solution/jian-zhi-offer-51-shu-zu-zhong-de-ni-xu-pvn2h/)

```python
class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        def merge_sort(l, r):
            if l >= r: return 0 # 说明到最里层
            m = (l + r) // 2
            res = merge_sort(l, m) + merge_sort(m + 1, r)
            i, j = l, m + 1
            tmp[l: r + 1] = nums[l: r + 1]
            for k in range(l, r + 1):
                if i == m + 1:
                    nums[k] = tmp[j]
                    j += 1
                elif j == r + 1 or tmp[i] <= tmp[j]:
                    nums[k] = tmp[i]
                    i += 1
                else:
                    nums[k] = tmp[j]
                    j += 1
                    res += m - i + 1 # 因为左边已经排好序，所以tmp[i] > tmp[j] 时，tmp[i]之后的数都满足 tmp[x] > tmp[j], where x > i 
            return res 
        
        tmp = [0] * len(nums)
        return merge_sort(0, len(nums) - 1) 
```

## 第 31 天 数学（困难）

[剑指 Offer 14- II. 剪绳子 II](https://leetcode-cn.com/problems/jian-sheng-zi-ii-lcof/)

```python
class Solution:
    def cuttingRope(self, n: int) -> int:
        if n == 2: return 1
        if n == 3: return 2
        if n % 3 == 0:
            return 3 ** (n // 3) % 1000000007
        if n % 3 == 1:
            return 3 ** (n // 3 - 1) * 4 % 1000000007
        if n % 3 == 2:
            return 3 ** (n // 3) * 2 % 1000000007
```

[剑指 Offer 43. 1～n 整数中 1 出现的次数](https://leetcode-cn.com/problems/1nzheng-shu-zhong-1chu-xian-de-ci-shu-lcof/)

输入一个整数 n ，求1～n这n个整数的十进制表示中1出现的次数。

例如，输入12，1～12这些整数中包含1 的数字有1、10、11和12，1一共出现了5次。


```python
class Solution:
    def countDigitOne(self, n: int) -> int:
        digit, res = 1, 0
        high, cur, low = n // 10, n % 10, 0
        while high != 0 or cur != 0:
            if cur == 0: res += high * digit
            elif cur == 1: res += high * digit + low + 1
            else: res += (high + 1) * digit
            low += cur * digit
            cur = high % 10
            high //= 10
            digit *= 10
        return res
```


[剑指 Offer 44. 数字序列中某一位的数字](https://leetcode-cn.com/problems/shu-zi-xu-lie-zhong-mou-yi-wei-de-shu-zi-lcof/)


```python
class Solution:
    def findNthDigit(self, n: int) -> int:
        digit, start, count = 1, 1, 9
        while n > count: # 1.
            n -= count
            start *= 10
            digit += 1
            count = 9 * start * digit
        num = start + (n - 1) // digit # 2.
        return int(str(num)[(n - 1) % digit]) # 3.
```

# 剑指 Offer 专项突击版

## 第 1 天 整数

## 第 2 天 整数

## 第 3 天 数组

[剑指 Offer II 007. 数组中和为 0 的三个数](https://leetcode-cn.com/problems/1fGaJU/)

给定一个包含 n 个整数的数组nums，判断nums中是否存在三个元素a ，b ，c ，使得a + b + c = 0 ？请找出所有和为 0 且不重复的三元组。

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        nums.sort()
        ans = list()
        
        for first in range(n):
            if first > 0 and nums[first] == nums[first - 1]:
                continue
            third = n - 1
            target = - nums[first]
            for second in range(first + 1, n):
                if second > first + 1 and nums[second] == nums[second - 1]:
                    continue
                while second < third and nums[second] + nums[third] > target:
                    third -= 1
                # 如果second和third指针重合，second再往右不会有符合条件的不重复组合
                if second == third:
                    break 
                if nums[second] + nums[third] == target: # 要写在退出条件之后，不然会加入second == third 的组合情况
                    ans.append([nums[first], nums[second], nums[third]])
        return ans 
```

[剑指 Offer II 008. 和大于等于 target 的最短子数组](https://leetcode-cn.com/problems/2VG8Kg/)

```python
class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        if not nums:
            return 0
        
        n = len(nums)
        ans = n + 1
        sums = [0]
        for i in range(n):
            sums.append(sums[-1] + nums[i])
        
        for i in range(1, n + 1):
            target = s + sums[i - 1]
            print(target)
            bound = bisect.bisect_left(sums, target)
            if bound != len(sums):
                ans = min(ans, bound - (i - 1))
        
        return 0 if ans == n + 1 else ans
```

[剑指 Offer II 009. 乘积小于 K 的子数组](https://leetcode-cn.com/problems/ZVAVXX/)

给定一个正整数数组 nums和整数 k ，请找出该数组内乘积小于 k 的连续的子数组的个数。

```python
class Solution:
    def numSubarrayProductLessThanK(self, nums, k):
        left = ans = 0
        total = 1
        for right, num in enumerate(nums):
            total *= num 
            while left <= right and total > k:
                total //= nums[left]
                left += 1
            if left <= right:
                ans += right - left + 1
        return ans 
```

## 第 4 天 数组

[剑指 Offer II 010. 和为 k 的子数组](https://leetcode-cn.com/problems/QTMn0o/)

给定一个整数数组和一个整数 k ，请找到该数组中和为 k 的连续子数组的个数。

[参考1](https://leetcode-cn.com/problems/QTMn0o/solution/he-wei-k-de-zi-shu-zu-by-leetcode-soluti-1169/) / 
[参考2](https://leetcode-cn.com/problems/QTMn0o/solution/pythonqian-zhui-he-by-zhsama-8lbz/)

```python
# 因为是仅看个数，所以记录某个前缀和出现的次数即可（因为中途可能有负数或者0）
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        count, pre = 0, 0
        d = dict()
        d[0] = 1
        for i in range(len(nums)):
            pre += nums[i] # 因为是从前往后遍历，无需list，只需常量存储
            if pre - k in d.keys():
                count += d[pre - k]
            d[pre] = d.get(pre, 0) + 1
        return count
        
```

[剑指 Offer II 011. 0 和 1 个数相同的子数组](https://leetcode-cn.com/problems/A1NYOS/)

给定一个二进制数组 nums , 找到含有相同数量的 0 和 1 的最长连续子数组，并返回该子数组的长度。


```python
class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        count = 0 # 累计的前缀和
        max_len = 0
        d = dict()
        n = len(nums)
        d[count] = -1 # 开始之前的前缀和为0
        for i in range(n):
            num = nums[i]
            if num == 1:
                count += 1
            else:
                count -= 1
            if count in d.keys():
                prev_index = d[count]
                max_len = max(max_len, i - prev_index)
            else:
                d[count] = i
            # [0,1,0,1,...] -> [(0),-1,0,-1,0] : d[0] = -1 且不再更新, 到了i=3则 3-(-1)=4
        return max_len

```

[剑指 Offer II 012. 左右两边子数组的和相等](https://leetcode-cn.com/problems/tvdfij/)

```python
class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        pre_sum, n, total = 0, len(nums), sum(nums)
        for i in range(n):
            if pre_sum == total - nums[i] - pre_sum:
                return i 
            pre_sum += nums[i]
        return -1
```

[剑指 Offer II 013. 二维子矩阵的和](https://leetcode-cn.com/problems/O4NDxx/)

```python
class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        self.m = matrix
        self.r, self.c = len(self.m), len(self.m[0])
        self.dp = [[0 for _ in range(self.c+1)] for _ in range(self.r+1)]
        self.dp[1][1] = self.m[0][0]
        for i in range(2, self.r+1):
            self.dp[i][1] = self.dp[i-1][1] + self.m[i-1][0]
        for j in range(2, self.c+1):
            self.dp[1][j] = self.dp[1][j-1] + self.m[0][j-1]
        for i in range(2, self.r+1):
            for j in range(2, self.c+1):
                self.dp[i][j] += self.dp[i-1][j] + sum(self.m[i-1][0:j])

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        return self.dp[row2+1][col2+1] - self.dp[row2+1][col1] - self.dp[row1][col2+1] + self.dp[row1][col1]

# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# param_1 = obj.sumRegion(row1,col1,row2,col2)
```

## 第 5 天 字符串

[剑指 Offer II 014. 字符串中的变位词](https://leetcode-cn.com/problems/MPnaiL/)

给定两个字符串 s1 和 s2，写一个函数来判断 s2 是否包含 s1 的某个变位词。

换句话说，第一个字符串的排列之一是第二个字符串的 子串 。

```python
import collections
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        # 抽象：在s2里的连续子串，含有的字符及对应个数（counter）和s1相同
        # (1) 暴力版本
        if len(s1)>len(s2):
            return False
        n1, n2 = len(s1), len(s2)
        # d1 = collections.Counter(s1)
        # for i, single in enumerate(s2):
        #     if single in d1.keys() and i+n1-1<len(s2):
        #         d2 = collections.Counter(s2[i: i+n1])
        #         print(d1, d2)
        #         if d2==d1:
        #             return True
        # return False
        # (2) 维护一个dict的版本，加减窗口里字符的个数
        d1 = collections.Counter(s1)
        if d1 == collections.Counter(s2[0:n1]):
            return True 
        for i in range(0, n1):
            if s2[i] in d1.keys():
                d1[s2[i]] -= 1
        for i in range(1, n2 - n1+1):
            if s2[i-1] in d1.keys():
                d1[s2[i-1]] += 1
            if s2[i+n1-1] in d1.keys():
                d1[s2[i+n1-1]] -= 1
            if set(s1) == set(s2[i:i+n1]) and set(d1.values())=={0}:
                return True 
        return False
```

[剑指 Offer II 015. 字符串中的所有变位词](https://leetcode-cn.com/problems/VabMRr/)

给定两个字符串 s 和 p，找到 s 中所有 p 的 变位词 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。

变位词 指字母相同，但排列不同的字符串。

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        if len(s)<len(p):
            return []
        ans = list()
        n1, n2 = len(p), len(s)
        d1 = collections.Counter(p)
        if d1 == collections.Counter(s[0:n1]):
            ans.append(0)
        print(d1)
        for i in range(0, n1):
            if s[i] in d1.keys():
                d1[s[i]] -= 1
        for i in range(1, n2 - n1+1):
            if s[i-1] in d1.keys():
                d1[s[i-1]] += 1
            if s[i+n1-1] in d1.keys():
                d1[s[i+n1-1]] -= 1
            if set(p) == set(s[i:i+n1]) and set(d1.values())=={0}:
                ans.append(i)
        return ans
```

[剑指 Offer II 016. 不含重复字符的最长子字符串](https://leetcode-cn.com/problems/wtcaE1/)

给定一个字符串 s ，请你找出其中不含有重复字符的 最长连续子字符串 的长度。

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # 双指针
        if len(s) <= 1: return len(s)
        if len(s)==2: return len(set(s))
        max_len = 1
        n = len(s)
        left, right = 0, 1
        while left < n - max_len:
            if s[right] in s[left: right]:
                left += 1     
            else:
                right += 1
            max_len = max(max_len, right - left)
        return max_len
```

## 第 6 天 字符串

[剑指 Offer II 017. 含有所有字符的最短字符串](https://leetcode-cn.com/problems/M1oyTv/)

给定两个字符串 s 和t 。返回 s 中包含 t 的所有字符的最短子字符串。如果 s 中不存在符合条件的子字符串，则返回空字符串 "" 。

如果 s 中存在多个符合条件的子字符串，返回任意一个。

- [参考](https://leetcode-cn.com/problems/M1oyTv/solution/python-shuang-zhi-zhen-ha-xi-biao-by-lau-xpyf/)

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if len(t) > len(s):
            return ""
        n_s, n_t = len(s), len(t)
        count = n_t # 记录达到的状态，=0时则满足
        ans = "" # return 答案
        d = Counter(t)
        tmp_len = n_s + 1
        left, right = 0, 0
        while right < n_s:
            # 1. 判断（右移后）最新一位的状态更新 2. 如果满足状态，更新ans，并且开始右移left-index 3. 判断left-index的右移后的状态改变
            if s[right] in d:
                if d[s[right]] > 0: 
                    count -= 1 # 只当t还有字符需要抵扣时才计算count
                d[s[right]] -= 1 # 即便是小于0也应该计算
            while count == 0: # 如果一直满足状态，说明left-index还需要右移，因为右边的字符串一直能满足条件
                if right - left + 1 < tmp_len:
                    tmp_len = right - left + 1
                    ans = s[left: right + 1]
                if s[left] in d:
                    if d[s[left]] == 0: # 刚好在加后大于0，说明出现实际缺补，才需要变动count
                        count += 1
                    d[s[left]] += 1
                left += 1
            right += 1
        return ans
```

[剑指 Offer II 018. 有效的回文](https://leetcode-cn.com/problems/XltzEq/)

给定一个字符串 s ，验证 s 是否是 回文串 ，只考虑字母和数字字符，可以忽略字母的大小写。

本题中，将空字符串定义为有效的 回文串。

```python
class Solution:
    def isPalindrome(self, s: str) -> bool:
        t = ''.join([i.lower() for i in s if i.isnumeric() or i.isalpha()])
        return True if t == t[::-1] else False
```

[剑指 Offer II 019. 最多删除一个字符得到回文](https://leetcode-cn.com/problems/RQku0D/)

给定一个非空字符串 s，请判断如果 最多 从字符串中删除一个字符能否得到一个回文字符串。

```python
class Solution:
    def validPalindrome(self, s: str) -> bool:
        # 双指针：（1）s[left]==s[right]：Y则left+=1，right-=1 N则判断s[left: right] 或者 s[left+1:right+1]是否符合回文（单字符也是回文）
        def zero2none(x):
            if x<0:
                return None
            else:
                return x
        if s == s[::-1]:
            return True
        left, right = 0, len(s) - 1
        flag = True
        while left < right:
            if s[left] == s[right]:
                left += 1
                right -= 1
            else:
                # print(left, right)
                # print(s[left: right], s[left + 1: right + 1])
                if s[left: right] == s[right - 1: zero2none(left - 1):  -1] or s[left + 1: right + 1] == s[right: zero2none(left) : -1] or right == left + 1:
                    return True
                else:
                    return False
        return True
```

[剑指 Offer II 020. 回文子字符串的个数](https://leetcode-cn.com/problems/a7VOhD/)

给定一个字符串 s ，请计算这个字符串中有多少个回文子字符串。

具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被视作不同的子串。

```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        # dp，由对角线填补
        n = len(s)
        dp = [[0 for _ in range(n)] for _ in range(n)]
        ans = 0
        for i in range(n): # 对角线
            dp[i][i] = 1
            ans += 1
        for i in range(n-1):
            dp[i][i+1] = 1 if s[i] == s[i+1] else 0
            ans += dp[i][i+1]
        for add in range(2, n):
            for i in range(n):
                j = i + add 
                # print(i, j)
                if j <= n - 1:                
                    dp[i][j] = 1 if dp[i+1][j-1] == 1 and s[i]==s[j] else 0
                    # print(i, j, dp[i][j])
                    ans += dp[i][j]
        # print(dp)
        return ans 
```

## 第 7 天 链表

[剑指 Offer II 021. 删除链表的倒数第 n 个结点](https://leetcode-cn.com/problems/SLwz0R/)

给定一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。

解法：双指针（快慢指针）、压入栈（先进后出，弹出的第n个则为倒数的第n个）



[剑指 Offer II 023. 两个链表的第一个重合节点](https://leetcode-cn.com/problems/3u1WK4/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        # 简单法
        # a, b = headA, headB
        # s = set()
        # while a:
        #     s.add(a)
        #     a = a.next 
        # while b:
        #     if b in s:
        #         return b 
        #     else:
        #         b = b.next 
        # return None

        # 双指针
        if not headA or not headB:
            return None 
        a, b = headA, headB
        while a!=b:
            a = a.next if a else headB
            b = b.next if b else headA
        return a 
```

## 第 8 天 链表

[剑指 Offer II 024. 反转链表](https://leetcode-cn.com/problems/UHnkqh/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if not head:
            return head 
        stack = list()
        p = head 
        while p:
            stack.append(p)
            p = p.next
        ans = stack[-1]
        while stack:
            p = stack.pop()
            p.next = stack[-1] if stack else None
        return ans
```

[剑指 Offer II 025. 链表中的两数相加](https://leetcode-cn.com/problems/lMSNwu/)

给定两个 非空链表 l1和 l2 来代表两个非负整数。数字最高位位于链表开始位置。它们的每个节点只存储一位数字。将这两数相加会返回一个新的链表。

可以假设除了数字 0 之外，这两个数字都不会以零开头。

解法
- 反转链表来对齐末位
- 通过栈来进行定位

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        if l1 == ListNode(0) or l2 == ListNode(0):
            return l1 if l2 == ListNode(0) else l2 
        n1, n2 = '', ''
        p1, p2 = l1, l2 
        while p1:
            n1 += str(p1.val)
            p1 = p1.next 
        while p2:
            n2 += str(p2.val)
            p2 = p2.next 
        n = int(n1) + int(n2)
        ans = ListNode(0)
        p = ans
        # print(n)
        for pos, i in enumerate(str(n)):
            p.val = int(i)
            if pos == len(str(n)) - 1:
                return ans
            p.next = ListNode()
            p = p.next
```

[剑指 Offer II 026. 重排链表](https://leetcode.cn/problems/LGjMqU/)

- 解题思路1：因为是头尾相接，所以利用两个栈以及弹出，重新构造顺序。注意边缘条件

执行用时：
64 ms
, 在所有 Python3 提交中击败了
98.69%
的用户
内存消耗：
23.4 MB
, 在所有 Python3 提交中击败了
12.99%
的用户

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        # 1. stack1 and stack2 to store 
        # 2. stack1.append() -> [1,2,3,4,5]
        # 3. stack1.pop() while time <= n // 2 -> stack1 = [1,2,3], stack2 [4,5]
        # 4. curr = stack1.pop() if len%2==1 else None 
        # 5. while s1 and s2: p1 = stack1.pop(), p2 = stack2.pop(), p1.next = p2, p2.next = curr, curr = p1 
        stack1, stack2 = list(), list()
        p = head 
        while p:
            stack1.append(p)
            p = p.next 
        n = len(stack1)
        while len(stack2) < n // 2:
            stack2.append(stack1.pop())
        curr = stack1.pop() if len(stack1) - len(stack2) == 1 else None 
        if curr:
            curr.next = None 
        while stack1 and stack2:
            p1 = stack1.pop()
            p2 = stack2.pop()
            p1.next = p2 
            p2.next = curr 
            curr = p1 
        
```

## 第 9 天 链表

[剑指 Offer II 027. 回文链表](https://leetcode.cn/problems/aMhZSa/)

给定一个链表的 头节点 head ，请判断其是否为回文链表。如果一个链表是回文，那么链表节点序列从前往后看和从后往前看是相同的。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        # new = ListNode()
        # p1 = new 
        q = head
        stack = list()
        while q:
            stack.append(q.val)
            q = q.next 
        return True if stack == stack[::-1] else False
```

[剑指 Offer II 028. 展平多级双向链表](https://leetcode.cn/problems/Qv1Da2/)

[参考](https://leetcode.cn/problems/Qv1Da2/solution/zhan-ping-duo-ji-shuang-xiang-lian-biao-x5ugr/)

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val, prev, next, child):
        self.val = val
        self.prev = prev
        self.next = next
        self.child = child
"""

class Solution:
    def flatten(self, head: 'Node') -> 'Node':
        def dfs(node):
            # 作用是返回node的最后一个位置，用于有child情况下，last作为next，而原本的next作为last的next
            curr = node 
            last = None 
            while curr:
                nxt = curr.next # eq0
                if curr.child:
                    # 将child转为next，找到末尾位置，接上原本的next，也就是目前的nxt（eq0）；再置空child
                    child_last = dfs(curr.child)
                    # 将 node 与 child 相连
                    curr.next = curr.child 
                    curr.child.prev = curr 

                    # 如果nxt不为空，将last和nxt连接
                    if nxt:
                        child_last.next = nxt 
                        nxt.prev = child_last
                    curr.child = None 
                    last = child_last
                else:
                    last = curr 
                curr = nxt # 往next遍历
            return last 
        dfs(head)
        return head
```

[剑指 Offer II 029. 排序的循环链表](https://leetcode.cn/problems/4ueAj6/)

给定循环单调非递减列表中的一个点，写一个函数向这个列表中插入一个新元素 insertVal ，使这个列表仍然是循环升序的。

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    def insert(self, head: 'Node', insertVal: int) -> 'Node':
        if not head:
            node = Node(insertVal)
            node.next = node 
            return node 
        p = head 
        while head != p.next: # 这个很重要，说明以下情况均不符合，比如 [3,3,3], 2
            if p.next.val < p.val: # [1,2,3] 里面 3的情况
                if insertVal > p.val: break
                if insertVal < p.next.val: break 
            if insertVal >= p.val and insertVal <= p.next.val:
                break 
            p = p.next 
        p.next = Node(insertVal, p.next)
        return head
```

## 第 10 天 哈希表

[剑指 Offer II 030. 插入、删除和随机访问都是 O(1) 的容器](https://leetcode.cn/problems/FortPu/)

```python
class RandomizedSet:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.d = dict()
        self.stack = list()

    def insert(self, val: int) -> bool:
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        """
        if val in self.d.keys():
            return False
        self.d[val] = len(self.stack)
        self.stack.append(val)
        return True 
         
    def remove(self, val: int) -> bool:
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        """
        if val not in self.d.keys():
            return False 
        self.d[self.stack[-1]] = self.d[val]

        self.stack[self.d[val]] = self.stack[-1]
        self.stack.pop()
        del self.d[val]
        return True 

    def getRandom(self) -> int:
        """
        Get a random element from the set.
        """
        pos = random.randint(0, len(self.stack) - 1)
        return self.stack[pos]

# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()
```

[剑指 Offer II 031. 最近最少使用缓存](https://leetcode.cn/problems/OrIXps/)


```python
import queue
class LRUCache:

    def __init__(self, capacity: int):
        self.n = capacity
        self.q = queue.deque() # 最左端为最久未使用，一使用即放到最右端
        self.d = dict()

    def get(self, key: int) -> int:
        if key in self.d.keys() and key in self.q:
            # 更新最久未之用的queue序列
            tmp = list(self.q)
            idx = tmp.index(key)
            tmp[idx:len(tmp)-1] = tmp[idx+1:]
            tmp[-1] = key 
            self.q = queue.deque(tmp)
        return self.d[key] if key in self.d.keys() else -1

    def put(self, key: int, value: int) -> None:
        if key in self.d.keys():
            tmp = list(self.q)
            idx = tmp.index(key)
            tmp[idx:len(tmp)-1] = tmp[idx+1:]
            tmp[-1] = key 
            self.q = queue.deque(tmp)
            self.d[key] = value 
        elif len(self.q) < self.n:
            self.q.append(key)
            self.d[key] = value 
        else:
            curr = self.q.popleft()
            if curr in self.d.keys():
                del self.d[curr]
            self.d[key] = value
            self.q.append(key)

# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```

[剑指 Offer II 032. 有效的变位词](https://leetcode.cn/problems/dKk3P7/)

```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        ds = collections.Counter(s)
        dt = collections.Counter(t)
        if ds == dt and s != t:
            return True 
        else:
            return False
```

## 第 12 天 栈

[剑指 Offer II 036. 后缀表达式](https://leetcode.cn/problems/8Zf90G/)

```python
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stack = list()
        for i in tokens:
            if i not in ['+','-','*','/']:
                stack.append(int(i))
            else:
                p1 = stack.pop()
                p2 = stack.pop()
                if i == '+': stack.append(p1 + p2)
                elif i == '-': stack.append(p2 - p1)
                elif i == '*': stack.append(p1 * p2)
                elif i == '/': stack.append(int(p2 / p1))
            # print(stack)
        return stack[-1]
```

[剑指 Offer II 037. 小行星碰撞](https://leetcode.cn/problems/XagZNi/)

```python
class Solution:
    def asteroidCollision(self, asteroids):
        s, p = [], 0
        while p < len(asteroids):
            if not s or s[-1] < 0 or asteroids[p] > 0:
                s.append(asteroids[p])
            elif s[-1] <= -asteroids[p]:
                if s.pop() < -asteroids[p]:
                    continue # 返回while
            print(s)
            p += 1
        return s
```

[剑指 Offer II 038. 每日温度](https://leetcode.cn/problems/iIQa4I/)

请根据每日 气温 列表 temperatures ，重新生成一个列表，要求其对应位置的输出为：要想观测到更高的气温，至少需要等待的天数。如果气温在这之后都不会升高，请在该位置用 0 来代替。

[参考](https://leetcode.cn/problems/iIQa4I/solution/mei-ri-wen-du-by-leetcode-solution-vh9j/)

- 逆序遍历得到比当前位置温度高的最短index，需要一个dict来存储。
复杂度 O(NM), M为温度范围, N为数组长度

- 栈来做，利用向后找更高气温的特点，形成反向思维。一旦有比目前(A)更大的数(B)，则弹出目前的数(A)并记录状态。由于比目前更大的数(B)更大的数(C)还未出现，所以弹出A是没问题的（因为对于A来说，能找B就不找C）。
  复杂度 O(N)

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        # （1）利用dict进行存储
        # n = len(temperatures)
        # ans, nxt, big = [0] * n, dict(), 10 ** 9 + 1
        # for i in range(n-1, -1, -1):
        #     warmer_index = min([nxt.get(tem, big) for tem in range(temperatures[i]+1, 102)])
        #     if warmer_index != big:
        #         ans[i] = warmer_index - i 
        #     nxt[temperatures[i]] = i # 此刻更新最新位置，相当于单独跑nxt
        # return ans 
        # ------------------------------------------------- 
        # （2）利用栈，存储每一次的最近高气温的index
        n = len(temperatures)
        ans, preIndex = [0] * n, list()
        for i, j in enumerate(temperatures):
            while preIndex and j > temperatures[preIndex[-1]]:
                tmp = preIndex.pop()
                ans[tmp] = i - tmp
            preIndex.append(i)
        return ans 
```


