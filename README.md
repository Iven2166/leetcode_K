# leetcode_K


题型
- 回溯：https://leetcode-cn.com/tag/backtracking/problemset/
- 字符串：https://leetcode-cn.com/tag/string/problemset/


```markdown 
-- https://markmap.js.org/repl
# 算法与数据结构-对应解法思考
## 字符串
### 双指针
### 哈希表
```

排序

原理参考：https://blog.csdn.net/pange1991/article/details/85460755

| 排序方法           | 平均时间  | 最好时间 | 最坏时间 | 参考  |
|--------------------|-----------|----------|----------|---|
| 快速排序(不稳定)   | O(nlogn)  | O(nlogn) | O(n^2)   | [参考-快排与归并](https://www.cnblogs.com/tuyang1129/p/12857821.html#:~:text=%E4%BA%86%E8%A7%A3%E5%BD%92%E5%B9%B6%E6%8E%92%E5%BA%8F%E7%9A%84%E5%BA%94%E8%AF%A5,%E5%BA%8F%E5%88%97%E4%B8%8D%E5%90%8C%E8%80%8C%E4%BA%A7%E7%94%9F%E6%B3%A2%E5%8A%A8%E3%80%82) |
| 归并排序(稳定)     | O(nlogn)  | O(nlogn) | O(nlogn) |   |
| 冒泡排序(稳定)     | O(n^2)    | O(n)     | O(n^2)   | [冒泡排序](https://blog.csdn.net/shengqianfeng/article/details/100016931#:~:text=%E5%86%92%E6%B3%A1%E6%8E%92%E5%BA%8F%E7%9A%84%E5%9F%BA%E6%9C%AC,%E5%B0%B1%E6%8A%8A%E5%AE%83%E4%BB%AC%E4%BA%A4%E6%8D%A2%E8%BF%87%E6%9D%A5%E3%80%82)  |
| 选择排序(不稳定)   | O(n^2)    | O(n^2)   | O(n^2)   |   |
| 堆排序(不稳定)     | O(nlogn)  | O(nlogn) | O(nlogn) |   |
| 直接插入排序(稳定) | O(n^2)    | O(n)     | O(n^2)   |   |
| 桶排序(不稳定)     | O(n)      | O(n)     | O(n)     |   |
| 基数排序(稳定)     | O(n)      | O(n)     | O(n)     |   |
| 希尔排序(不稳定)   | O(n^1.25) |          |          |   |


