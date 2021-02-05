# 1202

交换字符串中的元素

pairs可以构建一个并查集，同一个连通分量里的可以随意排序

把属于同一连通分量的字符拿出来分组排序，再放回去

排序的时候按字母序降序排，放回去的时候取连通分量的pop()即可

并查集 https://zhuanlan.zhihu.com/p/93647900/

python创建全1列表：

```list=[1]*n```

写成类运行会慢一些

```python
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        length=len(s)
        fa=[i for i in range(length)]
        rank=[1]*length

        def find(x):
            if(x == fa[x]):
                return x
            else:
                fa[x] = find(fa[x])  #父节点设为根节点
                return fa[x]         #返回父节点

        def merge(i, j):
            x = find(i)
            y = find(j)    #先找到两个根节点
            if (rank[x] <= rank[y]):
                fa[x] = y
            else:
                fa[y] = x

            if (rank[x] == rank[y] and x != y):
                rank[y]+=1  #如果深度相同且根节点不同，则新的根节点的深度+1

        for x, y in pairs:
            merge(x, y)
        
        group=[]
        for i in range(length):
            tmp=[]
            group.append(tmp)

        for i in range(length):
            ch=s[i]
            group[find(i)].append(ch)
        
        for i in range(length):
            group[i].sort(reverse=True)

        ans=''
        for i in range(length):
            x=find(i)
            ans+=group[x].pop()
        
        return ans
```



# 1

两数之和

解法一：哈希表，O(n)

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
            hashtable=dict()
            for i in range(len(nums)):
                num=nums[i]
                if target-num in hashtable:
                    return [i,hashtable[target-num]]
                else:
                    hashtable[num]=i
```

利用python的字典结构，写法比较简单

解法二：

先排序，然后从头尾往中间找，O(nlogn)

找到后再去顺序查找这两个数字出现的位置

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        sort_num=sorted(nums)

        l=0
        r=len(nums)-1

        while(l<r):
            s=sort_num[l]
            b=sort_num[r]
            if s+b==target:
                break
            elif s+b<target:
                l+=1
            else:
                r-=1
        
        x=None
        y=None
        for i in range(len(nums)):
            if x!=None and y!=None:
                break

            #同一个数字可能出现两次
            if nums[i]==s and x==None:
                x=i
            elif nums[i]==b:
                y=i
    
        return [x,y]
```



# 1203

项目管理

双重拓扑排序

将项目分到组，然后对组进行排序，再对组内的项目进行排序

对于没有组的项目，将其自己视为一个组

```python
class Solution:
    def topo_sort(self,items,indegree,neighbour):
        q=collections.deque()
        res=[]

        for item in items:
            if indegree[item]==0:
                q.append(item)
        
        while q:
            cur=q.popleft()
            res.append(cur)

            for ne in neighbour[cur]:
                indegree[ne]-=1
                if indegree[ne]==0:
                    q.append(ne)
        
        return res
        

    def sortItems(self, n: int, m: int, group: List[int], beforeItems: List[List[int]]) -> List[int]:
        max_group_id=m
        for i in range(n):
            if group[i]==-1: #没有组的项目
                group[i]=max_group_id
                max_group_id+=1
        
        belong=[[] for _ in range(max_group_id)]
        task_indegree=[0]*n
        task_neighbour=[[] for _ in range(n)]
        group_indegree=[0]*max_group_id
        group_neighbour=[[] for _ in range(max_group_id)]

        for i in range(n):
            belong[group[i]].append(i)

            for prev in beforeItems[i]:
                if group[prev]==group[i]: #前导项目在一个组，组内有拓扑关系
                    task_indegree[i]+=1
                    task_neighbour[prev].append(i)
                else: #前导项目不在一个组，组间有拓扑关系
                    group_indegree[group[i]]+=1
                    group_neighbour[group[prev]].append(group[i])
        
        #小组拓扑排序
        group_sort=self.topo_sort([i for i in range(max_group_id)],group_indegree,group_neighbour)

        res=[]

        if len(group_sort)!=max_group_id:
            return []

        for i in group_sort:
            #每组内拓扑排序
            task_sort=self.topo_sort(belong[i],task_indegree,task_neighbour)

            if len(task_sort)!=len(belong[i]):
                return []
            
            res.extend(task_sort)

        return res
```



# 9

回文数

将后一半的数字反转过来

简单情况：0、负数、10的倍数

如何知道反转数字的位数已经达到原始数字位数的一半：

奇数：x==rev/10

偶数：x==rev

原始数字小于等于反转后的数字

```python
class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x==0:
            return True
        elif x<0 or x%10==0:
            return False
        
        rev=0
        while(x>rev):
            rev=rev*10+x%10
            x=x//10

        if x==rev or x==(rev//10):
            return True
        else:
            return False
```



# 13

罗马数字转整数

分类判断

```python
class Solution:
    def romanToInt(self, s: str) -> int:
        ans=0
        i=0
        for i in range(len(s)):
            ch=s[i]
            if ch=='I':
                if i+1<len(s) and (s[i+1]=='V' or s[i+1]=='X'):
                    ans-=1
                else:
                    ans+=1
            elif ch=='V':
                ans+=5
            elif ch=='X':
                if i+1<len(s) and (s[i+1]=='L' or s[i+1]=='C'):
                        ans-=10
                else:
                    ans+=10
            elif ch=='L':
                ans+=50
            elif ch=='C':
                if i+1<len(s) and (s[i+1]=='D' or s[i+1]=='M'):
                        ans-=100
                else:
                    ans+=100
            elif ch=='D':
                ans+=500
            elif ch=='M':
                ans+=1000

        return ans
```



# 14

最长公共前缀

注意字符串为空的情况

```python
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        public=""
        i=0

        if len(strs)==0:
            return ""

        while(True):
            if i==len(strs[0]):
                return public
                
            ch=strs[0][i]
            for s in strs:
                if i==len(s) or s[i]!=ch:
                    return public
            
            public+=ch
            i+=1
```



# 20

有效的括号

经典括号栈

```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack=collections.deque()

        for b in s:
            if b=='(' or b=='[' or b=='{':
                stack.append(b)
            elif b==')':
                if len(stack)==0 or stack.pop()!='(':
                    return False
            elif b==']':
                if len(stack)==0 or stack.pop()!='[':
                    return False
            elif b=='}':
                if len(stack)==0 or stack.pop()!='{':
                    return False

        if len(stack)>0:
            return False

        return True
```



# 21

合并两个有序链表

指针操作

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        head=ListNode()

        tmp=head
        while l1 and l2:
            if l1.val<l2.val:
                tmp.next=l1
                l1=l1.next
            else:
                tmp.next=l2
                l2=l2.next
            tmp=tmp.next
        
        if l1:
            tmp.next=l1
        else:
            tmp.next=l2
        
        return head.next
```



# 26

删除排序数组中的重复项

两个指针，把下一个不同的放到该放到的位置

不用每一步都移动后面所有的

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        i=0
        for j in range(len(nums)):
            if nums[j]!=nums[i]:
                i+=1
                nums[i]=nums[j]
        
        return i+1
```



# 684

冗余连接

并查集

已经在同一连通子图里的边就是多余的

```python
class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        length=len(edges)+1
        fa=[i for i in range(length)]
        rank=[1]*length

        def find(x):
            if(x == fa[x]):
                return x
            else:
                fa[x] = find(fa[x])  
                return fa[x]         

        def merge(i, j):
            x = find(i)
            y = find(j)    
            if (rank[x] <= rank[y]):
                fa[x] = y
            else:
                fa[y] = x

            if (rank[x] == rank[y] and x != y):
                rank[y]+=1  

        for x, y in edges:
            if find(x)==find(y):
                if x<y:
                    ans=[x,y]
                else:
                    ans=[y,x]
            else:
                merge(x, y)
        
        return ans
```



# 27

移除元素

方法一：

双指针，类似26

```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        j=0
        for i in range(len(nums)):
            if nums[i]!=val:
                if i!=j:
                    nums[j]=nums[i]
                j+=1
        
        return jpy
```



方法二：

如果删除的元素很少时，方法一中会有不必要的元素移动

利用题目中数组元素顺序可以更改的条件，将当前元素与最后一个元素进行交换，并将数组大小减少1

```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        n=len(nums)
        i=0

        while(i<n):
            if nums[i]==val:
                nums[i]=nums[n-1]
                n-=1
            else:
                i+=1
                
        return n
```



# 28

实现strStr()

字符串匹配：KMP

```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        n=len(haystack)
        m=len(needle)

        if m==0:
            return 0
        if n==0:
            return -1
        
        ne=[0]*m
        ne[0]=-2
        if m>1:
            ne[1]=-1

        for i in range(2,m):
            j=ne[i-1]+1
            while needle[i-1]!=needle[j] and j>-1:
                j=ne[j]+1
            ne[i]=j

        j=0
        i=0
        start=-1

        while start==-1 and i<n:
            if needle[j]==haystack[i]:
                j+=1
                i+=1
            else:
                j=ne[j]+1
                if j==-1:
                    j=0
                    i+=1
            if j==m:
                start=i-m
        
        return start
```



# 1018

可被5整除的二进制前缀

优化：只保留模5的余数、用位运算代替*2

```python
class Solution:
    def prefixesDivBy5(self, A: List[int]) -> List[bool]:
        num=0
        ans=[]
        for i in A:
            num=((num<<1) + i) % 5
            ans.append(not num)
        
        return ans
```



# 35

搜索插入位置

二分查找

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        def binSearch(left,right):
            if left==right:
                if nums[left]>=target:
                    return left
                else:
                    return left+1
            
            middle=(left+right)//2
            if nums[middle]==target:
                return middle
            elif nums[middle]<target:
                return binSearch(middle+1,right)
            else:
                return binSearch(left,middle)
        
        return binSearch(0,len(nums)-1)
```



# 38

外观数列

找有多少个连续的数字

```python
class Solution:
    def countAndSay(self, n: int) -> str:
        if n==1:
            return "1"

        prev=self.countAndSay(n-1)

        ans=""
        count=1
        ch=prev[0]
        

        for i in range(1,len(prev)):
            if prev[i]==ch:
                count+=1
            else:
                ans+=str(count)
                ans+=ch
                count=1
                ch=prev[i]
        
        ans+=str(count)
        ans+=ch

        return ans
```



# 628

三个数的最大乘积

两种可能：最大的三个数，最小的两个数和最大的数

如果都是负数，也是最大的三个数

```python
class Solution:
    def maximumProduct(self, nums: List[int]) -> int:
        s=sorted(nums)

        return max(s[0]*s[1]*s[-1],s[-3]*s[-2]*s[-1])
```

可以不排序，直接线性扫描找到最小的两个数和最大的三个数，复杂度降到O(n)

```python
class Solution:
    def maximumProduct(self, nums: List[int]) -> int:
        max1=max2=max3=-1001
        min1=min2=1001

        for num in nums:
            if num>max1:
                max3=max2
                max2=max1
                max1=num
            elif num>max2:
                max3=max2
                max2=num
            elif num>max3:
                max3=num
            
            if num<min1:
                min2=min1
                min1=num
            elif num<min2:
                min2=num

        return max(min1*min2*max1, max1*max2*max3)
```



# 1584

连接所有点的最小费用

即最小生成树，Prim算法，找距离已有生成树距离最小的点加入

```python
class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        def dist(p1,p2):
            return abs(p1[0]-p2[0])+abs(p1[1]-p2[1])

        num=len(points)
        lowcost=[1e8] * num #维护剩余点到最小生成树的距离
        v=[-1] * num #维护点是否加入生成树

        #把第一个点加入生成树
        v[0]=0
        for i in range(1,num):
            lowcost[i]=dist(points[0],points[i])

        ans=0
        
        while(True):
            minid=-1
            min_dist=1e8
			
            #找距离生成树最近的点
            for i in range(1,num):
                if lowcost[i]<min_dist and v[i]!=0:
                    min_dist=lowcost[i]
                    minid=i

            if minid==-1: #全部都加入了
                return ans
            else:
                #加入生成树
                v[minid]=0
                ans+=min_dist

                for i in range(1,num):
                    if v[i]!=0: #更新还没有加入的点
                       lowcost[i]=min(lowcost[i],dist(points[minid],points[i]))
```



# 1489

找到最小生成树里的关键边和伪关键边

kruskal算法+枚举边

关键边：删去后：要么整个图不连通，不存在最小生成树；要么整个图连通，但是最小生成树的权值比最优的情况大

伪关键边：在某一最小生成树中出现，但不会出现在所有最小生成树中。在计算最小生成树时，最先考虑这条边，如果最小生成树的权值等于最优的情况，则为伪关键边

关键边也是伪关键边，所有应先判断关键边，如果不是再判断伪关键边

```python
class FindUnion: #并查集
    def __init__(self, length):
        self.fa=[i for i in range(length)]
        self.rank=[1]*length
        self.setcount=length #连通分量数量

    def find(self, x):
            if(x == self.fa[x]):
                return x
            else:
                self.fa[x] = self.find(self.fa[x])  #父节点设为根节点
                return self.fa[x]         #返回父节点

    def merge(self, i, j):
        x = self.find(i)
        y = self.find(j)    

        if x==y:
            return False

        if (self.rank[x] <= self.rank[y]):
            self.fa[x] = y
        else:
            self.fa[y] = x

        if (self.rank[x] == self.rank[y] and x != y):
            self.rank[y]+=1  #如果深度相同且根节点不同，则新的根节点的深度+1

        self.setcount-=1
        return True

class Solution:
    def findCriticalAndPseudoCriticalEdges(self, n: int, edges: List[List[int]]) -> List[List[int]]:
        length=len(edges)
        for i,edge in enumerate(edges):
            edge.append(i) #边的下标，方便给出答案

        edges.sort(key=lambda x:x[2]) #按边的权值排序

        #先算最小生成树
        uf_std=FindUnion(n)
        value=0 #最小的权值
        for i in range(length):
            if uf_std.merge(edges[i][0], edges[i][1]):
                value+=edges[i][2]

        key_edge=[]
        fkey_edge=[]

        for i in range(length):
            #判断关键边
            uf=FindUnion(n)
            v=0
            for j in range(length):
                if j!=i and uf.merge(edges[j][0],edges[j][1]):
                    v+=edges[j][2]
            
            if uf.setcount!=1 or (uf.setcount==1 and v>value):
                key_edge.append(edges[i][3])
                continue
            
            #判断伪关键边
            uf=FindUnion(n)
            uf.merge(edges[i][0],edges[i][1])
            v=edges[i][2]
            for j in range(length):
                if j!=i and uf.merge(edges[j][0],edges[j][1]):
                    v+=edges[j][2]

            if v==value:
                fkey_edge.append(edges[i][3])

        return [key_edge,fkey_edge]
```



# 1319

连通网络的操作次数

并查集，求有redund根多余的线，以及s+1个连通分量，说明有s个分量没有加入网络

只要一根线就可以使某一分量加入网络，所以判断s和redund的大小关系

```python
class Solution:
    def makeConnected(self, n: int, connections: List[List[int]]) -> int:
        fa=[i for i in range(n)]
        rank=[1]*n
        self.s=n-1

        def find(x):
            if(x == fa[x]):
                return x
            else:
                fa[x] = find(fa[x])  #父节点设为根节点
                return fa[x]         #返回父节点

        def merge(i, j):
            x = find(i)
            y = find(j)    #先找到两个根节点

            if x==y:
                return False

            if (rank[x] <= rank[y]):
                fa[x] = y
            else:
                fa[y] = x

            if (rank[x] == rank[y] and x != y):
                rank[y]+=1  
            
            self.s -= 1
            return True

        redund=0
        for connection in connections:
            if not merge(connection[0],connection[1]):
                redund+=1
        
        if self.s<=redund:
            return self.s
        else:
            return -1
```



# 989

数组形式的整数加法

直接把K加到数组的最后一位，再一点点往前做

需要考虑第一位有进位和K比较大的情况，要在列表前面插入

```python
class Solution:
    def addToArrayForm(self, A: List[int], K: int) -> List[int]:
        n=len(A)
        num=K

        i=n-1
        while i>=0:
            num += A[i] 
            A[i] = num % 10
            num = num // 10
            i-=1
        
        while num>0:
            A.insert(0, num % 10)
            num = num // 10

        return A
```



# 674

最长连续递增序列

经典顺序扫描，注意考虑为空的情况

```python
class Solution:
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        if len(nums)==0:
            return 0
            
        max_len=1
        current_len=1

        for i in range(1,len(nums)):
            if nums[i]>nums[i-1]:
                current_len+=1
            else:
                if current_len > max_len:
                    max_len = current_len

                current_len = 1
        
        if current_len > max_len:
                max_len = current_len
        
        return max_len
```



# 959

由斜杠划分区域

想不到吧这也是并查集

<img src="C:\Users\Yuxiang Lu\AppData\Roaming\Typora\typora-user-images\image-20210125150344319.png" alt="image-20210125150344319" style="zoom:67%;" />

将每个格子划分成四个小三角形，分别在**单元格内**和**单元格间**进行合并

**单元格内：**

空格：合并0、1、2、3；

/：合并0、3，合并1、2；

\：合并0、1，合并2，3

**单元格间：**

<img src="C:\Users\Yuxiang Lu\AppData\Roaming\Typora\typora-user-images\image-20210125150610427.png" alt="image-20210125150610427" style="zoom:67%;" />

需要向右、向下尝试合并

向右：合并1（当前格）和3（右边1列的格）红色部分

向下：合并2（当前格）和0（下边1行的格）蓝色部分

```python
class Solution:
    def regionsBySlashes(self, grid: List[str]) -> int:
        n=len(grid)
        fa=[i for i in range(4*n*n)]
        rank=[1]*(4*n*n)
        self.count=4*n*n #总共4*n*n个小三角形

        def find(x):
            if(x == fa[x]):
                return x
            else:
                fa[x] = find(fa[x])  #父节点设为根节点
                return fa[x]         #返回父节点

        def merge(i, j):
            x = find(i)
            y = find(j)    #先找到两个根节点
            if x==y:
                return 
            
            if (rank[x] <= rank[y]):
                fa[x] = y
            else:
                fa[y] = x

            if (rank[x] == rank[y] and x != y):
                rank[y]+=1  #如果深度相同且根节点不同，则新的根节点的深度+1

            self.count-=1
        
        for i in range(n):
            for j in range(n):
                order=n*i+j #当前格的编号

                if grid[i][j]=='/':
                    merge(4*order,4*order+3)
                    merge(4*order+1,4*order+2)
                elif grid[i][j]=='\\':
                    merge(4*order,4*order+1)
                    merge(4*order+2,4*order+3)
                else:
                    merge(4*order,4*order+1)
                    merge(4*order,4*order+2)
                    merge(4*order,4*order+3)
                
                if j<n-1:
                    merge(4*order+1,4*(order+1)+3)
                
                if i<n-1:
                    merge(4*order+2,4*(order+n))
        
        return self.count
```



# 721 

账户合并

又又又是并查集

因为邮箱地址难以操作，首先用哈希表```email2index```把邮箱地址映射到一个编号，用另一个哈希表```email2name```记录每个邮箱地址对应的名称

然后用并查集将邮箱地址进行合并，对于用一个连通分量中的邮箱，其属于一个账户，将其汇总到一个字典```index2email```中

```python
class FindUnion: #并查集
    def __init__(self, length):
        self.fa=[i for i in range(length)]
        self.rank=[1]*length

    def find(self, x):
            if(x == self.fa[x]):
                return x
            else:
                self.fa[x] = self.find(self.fa[x])  #父节点设为根节点
                return self.fa[x]         #返回父节点

    def merge(self, i, j):
        x = self.find(i)
        y = self.find(j)    

        if x==y:
            return

        if (self.rank[x] <= self.rank[y]):
            self.fa[x] = y
        else:
            self.fa[y] = x

        if (self.rank[x] == self.rank[y] and x != y):
            self.rank[y]+=1  #如果深度相同且根节点不同，则新的根节点的深度+1

class Solution:
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        email2index={}
        email2name={}

        i=0
        for account in accounts:
            for email in account[1:]:
                if not email in email2index:
                    email2index[email]=i #顺序安排编号
                    email2name[email]=account[0]
                    i+=1
        
        uf=FindUnion(len(email2index))

        for account in accounts:
            first_index=email2index[account[1]]
            for email in account[2:]:
                uf.merge(first_index,email2index[email])
        
        index2email=collections.defaultdict(list)
        for email,index in email2index.items():
            index=uf.find(index)
            index2email[index].append(email) #汇总到父节点中
        
        ans=[]
        for emails in index2email.values():
            a=[]
            a.append(email2name[emails[0]])
            a.extend(sorted(emails)) #记得排序
            ans.append(a)
        
        return ans
```



# 1232

缀点成线

简单数学方法
$$
(y_n-y_0)/(x_n-x_0)=(y_1-y_0)/(x_1-x_0)
$$
避免除法，将其对角相乘

```python
class Solution:
    def checkStraightLine(self, coordinates: List[List[int]]) -> bool:
        x0=coordinates[0][0]-coordinates[1][0]
        y0=coordinates[0][1]-coordinates[1][1]

        for coord in coordinates[2:]:
            x=coord[0]-coordinates[1][0]
            y=coord[1]-coordinates[1][1]

            if not x0*y==x*y0:
                return False
        
        return True
```



# 1128

等价多米诺骨牌对的数量

将每一对数字排成前者小于等于后者

为了方便表示，(x,y)=10*x+y，因为x和y都是一个数字

每一对数字都和已有的这一对数字形成一对新的骨牌对

```python
class Solution:
    def numEquivDominoPairs(self, dominoes: List[List[int]]) -> int:
        do_list=[0]*100
        ans=0
        
        for x,y in dominoes:
            if x<=y:
                a=10*x+y
            else:
                a=10*y+x
            
            ans+=do_list[a]
            do_list[a]+=1
        
        return ans
```



# 53

最大子序和

分治法

每个子区间$[l,r]$维护四个值：

```lsum``` 以$l$为左端点的最大子序和

```rsum``` 以$r$为右端点的最大子序和

```msum``` 区间内的最大子序和

```isum``` 区间和

合并：

```isum```： 左```isum```+右```isum```

```lsum```：max(左```lsum```, 左```isum```+右```lsum```）

```rsum```：max(右```rsum```, 左```rsum```+右```isum```）

```msum```：max(左```msum```, 右```msum```,  左```rsum```+右```lsum```）

答案为$[0,n-1]$的```msum```

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        def get(a,l,r):
            if l==r:
                return [a[l],a[l],a[l],a[l]]
            
            m=(l+r)//2
            lsub=get(a,l,m)
            rsub=get(a,m+1,r)

            isum=lsub[0]+rsub[0]
            lsum=max(lsub[1],lsub[0]+rsub[1])
            rsum=max(rsub[2],lsub[2]+rsub[0])
            msum=max(max(lsub[3],rsub[3]),lsub[2]+rsub[1])

            return [isum,lsum,rsum,msum]
        
        ans=get(nums,0,len(nums)-1)
        return ans[3]
```



dp方法：

$f[i]=max\{f[i-1]+a_i, a_i\}$

答案是$max_{0\le i\le n-1 }\{f[i]\}$

因为$f[i]$只用一次，所以用一个变量记录最大值即可，空间复杂性可以降到O(1)

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        pre=0
        ans=nums[0]
        for num in nums:
            pre=max(pre+num,num)
            ans=max(ans,pre)
        
        return ans
```



# 1579

保证图可完全遍历

为了保留最少数量的边，公共边是首要的

维护两个并查集，首先添加公共边，然后Alice和Bob分别添加自己的独占边，如果最后两个并查集都只包含一个连通分量，则说明两者都可以完全遍历整个图

在并查集进行合并时，每遇到一次失败的合并，即为一条多余的边

```python
class UnionFind:
    def __init__(self,length):
        self.fa=[i for i in range(length)]
        self.rank=[1]*length
        self.setcount=length

    def find(self,x):
        if x==self.fa[x]:
            return x
        self.fa[x]=self.find(self.fa[x])
        return self.fa[x]
    
    def union(self,i,j):
        x=self.find(i)
        y=self.find(j)

        if x==y:
            return False
        
        if self.rank[x]<=self.rank[y]:
            self.fa[x]=y
        else:
            self.fa[y]=x
        
        if self.rank[x]==self.rank[y]:
            self.rank[y]+=1
        
        self.setcount-=1
        return True

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        ufa=UnionFind(n)
        ufb=UnionFind(n)
        ans=0

        #每个点的编号从0开始
        for edge in edges:
            edge[1]-=1
            edge[2]-=1
        
        for t,u,v in edges:
            if t==3: #公共边
                if not ufa.union(u,v):
                    ans+=1
                else:
                    ufb.union(u,v)
            
        for t,u,v in edges:
            if t==1: #Alice独占边
                if not ufa.union(u,v):
                    ans+=1
            
            if t==2: #Bob独占边
                if not ufb.union(u,v):
                    ans+=1
            
        if ufa.setcount!=1 or ufb.setcount!=1:
            return -1
         
        return ans
```



# 724

寻找数组的中间索引

计算左边的和以及右边的和

注意只有一个元素时，一定是中间索引

```python
class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        lsum=0
        rsum=0
        for i in range(1,len(nums)):
            rsum+=nums[i]

        for i in range(0,len(nums)):
            if lsum==rsum:
                return i
            
            if i<len(nums)-1:
                lsum+=nums[i]
                rsum-=nums[i+1]
        
        return -1
```



# 839

相似字符串组

又又又是并查集

相似关系没有连通性，但是相似字符串组有连通性，所以需要枚举每一对字符串来判断相似性

Tricks：

剪枝：两个字符串已经在同一个字符串组里了

判断相似性：因为字符串都是彼此的字母异位词，所以只要看有几个位置的字母不一样，如果是0或2个就相似

```python
class Solution:
    def numSimilarGroups(self, strs: List[str]) -> int:
        length=len(strs)
        m=len(strs[0])
        fa=[i for i in range(length)]
        rank=[1]*length
        self.setcount=length

        def find(x):
            if(x == fa[x]):
                return x
            else:
                fa[x] = find(fa[x])  #父节点设为根节点
                return fa[x]         #返回父节点

        def merge(i, j):
            x = find(i)
            y = find(j)    #先找到两个根节点
            if (rank[x] <= rank[y]):
                fa[x] = y
            else:
                fa[y] = x

            if (rank[x] == rank[y] and x != y):
                rank[y]+=1  #如果深度相同且根节点不同，则新的根节点的深度+1
            
            self.setcount-=1
        
        def check(a,b):
            diff=0
            for i in range(m):
                if a[i]!=b[i]:
                    diff+=1
                    if diff>2:
                        return False
            
            return True

        
        for i in range(length):
            for j in range(i+1,length):
                fi=find(i)
                fj=find(j)

                if fi!=fj and check(strs[i],strs[j]):
                    merge(i,j)
        
        return self.setcount
```



# 101

对称二叉树

递归解法：如果一个树的左子树和右子树镜像对称，那么这个树是对称的

两个树互为镜像：根结点的值相同，每个树的左子树和另一个树的右子树镜像对称

```python
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        def check(a:TreeNode ,b:TreeNode):
            if (not a) and (not b):
                return True
            
            if (not a) or (not b):
                return False
            
            return a.val==b.val and check(a.left,b.right) and check(a.right,b.left)
        
        return check(root,root)
```

迭代解法：bfs遍历，根结点入队两次，每次提取两个结点比较，然后将两个结点的左右子结点按相反的顺序加入队列

```python
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        q=collections.deque()

        q.append(root)
        q.append(root)

        while len(q):
            a=q.popleft()
            b=q.popleft()

            if (not a) and (not b):
                continue
            
            if (not a) or (not b) or (not a.val==b.val):
                return False
            
            q.append(a.left)
            q.append(b.right)

            q.append(a.right)
            q.append(b.left)

        return True
```



# 888

公平的糖果棒交换

easyyyy

```python
class Solution:
    def fairCandySwap(self, A: List[int], B: List[int]) -> List[int]:
        suma=0
        sumb=0

        for i in range(len(A)):
            suma+=A[i]
        
        for i in range(len(B)):
            sumb+=B[i]
        
        diff=(sumb-suma)//2
        
        for i in range(len(A)):
            if (A[i]+diff) in B:
                return [A[i],A[i]+diff]
```



# 104

二叉树的最大深度

递归解法

```python
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        return max(self.maxDepth(root.left),self.maxDepth(root.right))+1
```

遍历解法

bfs遍历，一层一层地遍历

```python
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        q=collections.deque()
        q.append(root)

        ans=0
        while len(q):
            sz=len(q)

            //遍历一层
            while sz:
                node=q.popleft()
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
                sz-=1

            ans+=1//层数加一
        
        return ans
```

# 424

替换后的最长重复字符

双指针滑动窗口

右边界先尽可能右移，不能满足时停止，然后左边界右移一格（这时就满足了），再尝试右移右边界

```maxcount```维护的是所有窗口内最多出现的字符次数

建议再看看题解

```python
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        length=len(s)

        left=0
        right=0
        maxcount=0
        res=0
        freq={}

        while right<length:
            ch=s[right]
            if ch in freq.keys(): #窗口内字符出现的次数
                freq[ch]+=1
            else:
                freq[ch]=1
            
            
            maxcount=max(maxcount,freq[ch])
            right+=1
        
            if right-left>maxcount+k: #窗口不满足条件
                freq[s[left]]-=1 #左边界所在的字符出现次数减一
                left+=1 #左边界右移
        
            res=max(res,right-left)
        
        return res
```



# 121

买卖股票的最佳时机

只需遍历一次，对于每一天，考虑这一天之前的最小值以及能获得的利润

因为最小值是动态更新的，所以只需遍历一次

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        m=1e5
        pro=0
        
        for price in prices:
            if price<m:
                m=price    
            elif price-m>pro:
                pro=price-m
        
        return pro
```



# 136

只出现一次的数字

神奇异或

a^0=a

a^a=0

异或满足交换律和结合律

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        ans=0

        for num in nums:
            ans=ans^num
        
        return ans
```



# 643

子数组最大平均数

大小为k的滑动窗口

```python
class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        ans=0
        for i in range(k):
            ans+=nums[i]

        ne=k
        now=ans

        while ne<len(nums):
            now-=nums[ne-k]
            now+=nums[ne]
            ans=max(ans,now)
            ne+=1
        
        return ans/k
```



# 480

滑动窗口中位数

我的方法：半死做，中位数不变的情况就不用重新算

```python
class Solution:
    def medianSlidingWindow(self, nums: List[int], k: int) -> List[float]:
        left=0
        right=k
        ans=[]

        window=sorted(nums[left:right])
        if k%2==0:
            pre1=window[k//2-1]
            pre2=window[k//2]
            mid=(pre1+pre2)/2
        else:
            pre1=window[k//2]
            mid=pre1
        ans.append(mid)

        while right<len(nums):
            delete=nums[left]
            add=nums[right]
            left+=1
            right+=1

            if k%2==0:
                if not (delete<pre1 and add<pre1) or (delete>pre2 and add>pre2):#删除和添加的同时比两个都小或比两个都大
                    window=sorted(nums[left:right])
                    pre1=window[k//2-1]
                    pre2=window[k//2]
                    mid=(pre1+pre2)/2
            else:
                if not (delete<pre1 and add<pre1) or (delete>pre1 and add>pre1):#删除和添加的同时比前一个中位数小或大
                    window=sorted(nums[left:right])
                    pre1=window[k//2]
                    mid=pre1
            
            ans.append(mid)
            
        return ans
```

双堆+延迟删除

建议看题解，细节较多

```python
class DualHeap:
    def __init__(self, k: int):
        # 大根堆，维护较小的一半元素，注意 python 没有大根堆，需要将所有元素取相反数并使用小根堆
        self.small = list()
        # 小根堆，维护较大的一半元素
        self.large = list()
        # 哈希表，记录「延迟删除」的元素，key 为元素，value 为需要删除的次数
        self.delayed = collections.Counter()

        self.k = k
        # small 和 large 当前包含的元素个数，需要扣除被「延迟删除」的元素
        self.smallSize = 0
        self.largeSize = 0


    # 不断地弹出 heap 的堆顶元素，并且更新哈希表
    def prune(self, heap: List[int]):
        while heap:
            num = heap[0]
            if heap is self.small:
                num = -num
            if num in self.delayed:
                self.delayed[num] -= 1
                if self.delayed[num] == 0:
                    self.delayed.pop(num)
                heapq.heappop(heap)
            else:
                break
    
    # 调整 small 和 large 中的元素个数，使得二者的元素个数满足要求
    def makeBalance(self):
        if self.smallSize > self.largeSize + 1:
            # small 比 large 元素多 2 个
            heapq.heappush(self.large, -self.small[0])
            heapq.heappop(self.small)
            self.smallSize -= 1
            self.largeSize += 1
            # small 堆顶元素被移除，需要进行 prune
            self.prune(self.small)
        elif self.smallSize < self.largeSize:
            # large 比 small 元素多 1 个
            heapq.heappush(self.small, -self.large[0])
            heapq.heappop(self.large)
            self.smallSize += 1
            self.largeSize -= 1
            # large 堆顶元素被移除，需要进行 prune
            self.prune(self.large)

    def insert(self, num: int):
        if not self.small or num <= -self.small[0]:
            heapq.heappush(self.small, -num)
            self.smallSize += 1
        else:
            heapq.heappush(self.large, num)
            self.largeSize += 1
        self.makeBalance()

    def erase(self, num: int):
        self.delayed[num] += 1
        if num <= -self.small[0]:
            self.smallSize -= 1
            if num == -self.small[0]:
                self.prune(self.small)
        else:
            self.largeSize -= 1
            if num == self.large[0]:
                self.prune(self.large)
        self.makeBalance()

    def getMedian(self) -> float:
        return float(-self.small[0]) if self.k % 2 == 1 else (-self.small[0] + self.large[0]) / 2


class Solution:
    def medianSlidingWindow(self, nums: List[int], k: int) -> List[float]:
        dh = DualHeap(k)
        for num in nums[:k]:
            dh.insert(num)
        
        ans = [dh.getMedian()]
        for i in range(k, len(nums)):
            dh.insert(nums[i])
            dh.erase(nums[i - k])
            ans.append(dh.getMedian())
        
        return ans
```



# 1208

尽可能使字符串相等

两个字串的位置是相同的，所以先算出每个位置上的开销```diff```

然后用滑动窗口求出在最大预算内```diff```的最大子串和

```ord()```可以获取字符的ascii码

```python
class Solution:
    def equalSubstring(self, s: str, t: str, maxCost: int) -> int:
        diff=[]
        for i in range(len(s)):
            diff.append(abs(ord(s[i])-ord(t[i])))
        
        left=0
        right=0
        ans=0
        sum_=0

        while right<len(s):
            sum_+=diff[right]
            while sum_>maxCost:
                ans=max(ans,right-left)#不包含right
                sum_-=diff[left]
                left+=1
            
            ans=max(ans,right-left+1)#包含right
            right+=1
        
        return ans
```



# 141

环形链表

要求O(1)的空间复杂度

快慢指针，类似于龟兔赛跑

慢指针每次移动一个，快指针每次移动两个

如果链表有环，则快指针会超过慢指针一圈，即两者相遇，否则两者不会相遇

```python
class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        if (not head) or (not head.next):
            return False
        
        slow=head
        fast=head.next

        while slow != fast:
            if (not fast) or (not fast.next):
                return False
            
            slow=slow.next
            fast=fast.next.next

        return True
```



# 155

最小栈

辅助栈方法：元素入栈时，把当前栈的最小值存在辅助栈中

栈中最小值就是辅助栈的栈顶元素

```python
class MinStack:
    def __init__(self):
        self.stack=[]
        self.min_stack=[math.inf]

    def push(self, x: int) -> None:
        self.stack.append(x)
        self.min_stack.append(min(self.min_stack[-1],x))

    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]
```

不适用额外空间：

栈中保存和当前最小值的差值，```min_value```为当前栈的最小值

```python
class MinStack:
    def __init__(self):
        self.stack=[]
        self.min_value=-1

    def push(self, x: int) -> None:
        if not self.stack:
            self.stack.append(0)
            self.min_value=x
        else:
            diff=x-self.min_value
            self.stack.append(diff)
            if diff<0: #最小值有更新
                self.min_value=x

    def pop(self) -> None:
        diff=self.stack.pop()
        if diff<0: #最小值被pop
            top=self.min_value
            self.min_value=top-diff #上一个最小值
        else:
            top=self.min_value+diff
        return top

    def top(self) -> int:
        if self.stack[-1]<0:
            return self.min_value  
        else:
            return self.stack[-1] + self.min_value

    def getMin(self) -> int:
        return self.min_value
```

