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



# 1423

可获得的最大点数

滑动窗口，从最左k个反向滑到最右k个

```python
class Solution:
    def maxScore(self, cardPoints: List[int], k: int) -> int:
        sum_=0
        n=len(cardPoints)

        for i in range(k):
            sum_+=cardPoints[i]
        
        ans=sum_
        for i in range(k):
            sum_-=cardPoints[k-1-i]
            sum_+=cardPoints[n-1-i]
            ans=max(ans,sum_)
        
        return ans
```



# 160

相交链表

特别巧妙的双指针法

两个指针从两个链表头开始遍历，当pA到达尾部时，下一个指向B的头结点，pB到达尾部时，下一个指向A的头结点

如果pA和pB相遇，则该位置为相交结点，因为两个指针走过的距离是一样的

如果两个链表不相交，则其尾结点不同，检查即可

```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        if not headA or not headB:
            return None
            
        a=headA
        b=headB
        endA=None
        endB=None

        while a!=b:
            if not a.next:
                endA=a
                a=headB #重定位
            else:
                a=a.next

            if not b.next:
                endB=b
                b=headA #重定位
            else:
                b=b.next
			
            #检查尾结点
            if endA and endB and endA!=endB:
                return None
            
        return a
```



# 665

非递减序列

只能出现一次递减的情况

如果nums[0]>nums[1]，那么令nums[0]=nums[1]即可，并不会影响到后面，所以不用实际修改

当nums[i+1]<nums[i]时，出现递减的情况，又可分为两种情况：

（1）nums[i+1]>=nums[i-1]，令nums[i]=nums[i-1]即可，不影响后面，不用实际修改；

（2）nums[i+1]<nums[i-1]，要令nums[i+1]=nums[i]，会影响后面的判断，需要实际修改。

```python
class Solution:
    def checkPossibility(self, nums: List[int]) -> bool:
        flag=False
        for i in range(0,len(nums)-1):
            if nums[i]>nums[i+1]:
                if flag: #已经出现过一次
                    return False
                flag=True

                if i>0 and nums[i+1]<nums[i-1]:
                    nums[i+1]=nums[i]
                
        return True
```



# 978

最长湍流子数组

滑动窗口找最大长度

用asc表示前两个元素的大小情况，arr[i]>arr[i+1]时asc=1，arr[i]<arr[i+1]时asc=2，相等时为0

如果两个相等，要从下一个开始判断，即当前长度为1

```python
class Solution:
    def maxTurbulenceSize(self, arr: List[int]) -> int:
        if len(arr)==1:
            return 1

        asc=0
        ans=1
        now_len=1
        for i in range(0,len(arr)-1):
            if asc==1:
                if arr[i]<arr[i+1]:
                    now_len+=1
                    asc=2
                elif arr[i]>arr[i+1]:#不满足，重开
                    now_len=2
                else:
                    asc=0
            elif asc==2:
                if arr[i]>arr[i+1]:
                    now_len+=1
                    asc=1
                elif arr[i]<arr[i+1]:#不满足，重开
                    now_len=2
                else:
                    asc=0
            else: #初始化也放在这里
                if arr[i]>arr[i+1]:
                    asc=1
                    now_len=2
                elif arr[i]<arr[i+1]:
                    asc=2
                    now_len=2

            ans=max(ans,now_len)
        
        return ans
```



# 703

数据流中的第k大元素

用一个大小为k的最小堆来存储前k大的元素，堆顶就是当前第k大的元素

插入的时候，如果堆的大小大于k，就将堆顶元素弹出

```python
class KthLargest:
    def __init__(self, k: int, nums: List[int]):
        self.heap=Heap()
        self.k=k
        for num in nums:
            self.heap.push(num)
            if self.heap.size()>k:
                self.heap.pop()
        
    def add(self, val: int) -> int:
        self.heap.push(val)
        if self.heap.size()>self.k:
            self.heap.pop()
        return self.heap.top()

class Heap:
    def __init__(self):
        self.heap=[]
    
    def size(self):
        return len(self.heap)
    
    def top(self):
        if self.size:
            return self.heap[0]
        else:
            return None
    
    def push(self,a):
        """
        添加元素
        第一步，把元素加入堆的最后
        第二步，向上交换
        """
        self.heap.append(a)
        self.swap_up(self.size()-1)
    
    def pop(self):
        """
        弹出堆顶
        第一步，记录堆顶元素的值
        第二步，交换堆顶元素与末尾元素
        第三步，删除末尾元素
        第四步，新的堆顶元素向下交换
        """
        p=self.heap[0]
        self.heap[0],self.heap[self.size()-1]=self.heap[self.size()-1],self.heap[0]
        self.heap.pop()
        self.swap_down(0)
        return p
    
    def swap_up(self,index):
        """
        向上交换
        如果父节点和当前节点满足交换的关系（小顶堆是父节点元素更大），
        则持续将当前节点向上交换
        """
        while index:
            father=(index-1)//2

            if self.heap[father]<=self.heap[index]:
                break
            
            self.heap[father],self.heap[index]=self.heap[index],self.heap[father]
            index=father
    
    def swap_down(self,index):
        """
        向下交换
        如果子节点和当前节点满足交换的关系（小顶堆是子节点元素更小），
        则持续将当前节点向下交换
        """
        son=index*2+1
        while son<self.size():
            #选子节点中较小的那个
            if son+1<self.size() and self.heap[son+1]<self.heap[son]:
                son+=1
            
            if self.heap[index]<=self.heap[son]:
                break
            
            self.heap[son],self.heap[index]=self.heap[index],self.heap[son]
            index=son
            son=index*2+1
```



# 765

情侣牵手

并查集：如果第i对和第j对坐在一起，就把i和j连起来，代表要交换位置

一个连通分量里的可以形成一个环，沿环的方向交换

每个连通分量大小减1就是要交换的次数

```python
class Solution:
    def minSwapsCouples(self, row: List[int]) -> int:
        cou=len(row)//2
        fa=[i for i in range(cou)]
        rank=[1]*cou

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
        
        for i in range(cou):
            l=row[2*i]
            r=row[2*i+1]
            merge(l//2,r//2)

        count={}
        for i in range(cou):
            f=find(i)
            if f in count:
                count[f]+=1
            else:
                count[f]=1
 
        ans=0
        for num in count.values():
            ans+=num-1

        return ans 
```



# 485

最大连续1的个数

简简单单的遍历

```python
class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        ans=0
        now=0
        
        for i in range(len(nums)):
            if nums[i]==1:
                now+=1
            else:
                ans=max(ans,now)
                now=0
        
        ans=max(ans,now)
        return ans
```



# 561

数组拆分I

code很简单，主要是证明

![image-20210216123107172](C:\Users\Yuxiang Lu\AppData\Roaming\Typora\typora-user-images\image-20210216123107172.png)

```python
class Solution:
    def arrayPairSum(self, nums: List[int]) -> int:
        nums=sorted(nums)
        ans=0
        for i in range(0,len(nums),2):
            ans+=nums[i]
        return ans
```



# 566

重塑矩阵

方法一：一个一个遍历，从旧的放到新的

```python
class Solution:
    def matrixReshape(self, nums: List[List[int]], r: int, c: int) -> List[List[int]]:
        old_r=len(nums)
        old_c=len(nums[0])

        if not old_r*old_c==r*c:
            return nums

        m=0
        n=0
        ans=[]
        for i in range(r):
            ans_r=[]
            for j in range(c):
                ans_r.append(nums[m][n])
                if n==old_c-1:
                    m+=1
                    n=0
                else:
                    n+=1
            
            ans.append(ans_r)
        
        return ans
```

方法二：视为一维数组，直接算下标

```python
class Solution:
    def matrixReshape(self, nums: List[List[int]], r: int, c: int) -> List[List[int]]:
        old_r=len(nums)
        old_c=len(nums[0])

        if not old_r*old_c==r*c:
            return nums

        ans=[[0]*c for _ in range(r)]
        for i in range(r*c):
            ans[i//c][i%c]=nums[i//old_c][i%old_c]
        
        return ans
```



# 448

找到所有数组中消失的数字

把自己当作哈希表，原地修改

如果一个数字x出现过，就把nums[x]加上n

因为x可能已经被加过n，所以要先mod一下

最后找出nums[i]<=n的i

```python
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        n=len(nums)
        for num in nums:
            x=(num-1)%n
            nums[x]+=n
        
        ans=[]
        for i in range(n):
            if nums[i]<=n:
                ans.append(i+1)
        
        return ans
```



# 119

杨辉三角II

dp，空间复杂性O(k)，因为会覆盖前面的值，所以每一行中要从后往前计算新的

```python
class Solution:
    def getRow(self, rowIndex: int) -> List[int]:
        ans=[1]*(rowIndex+1)

        for row in range(1,rowIndex+1):
            for i in range(row-1,0,-1):
                ans[i]+=ans[i-1]
            
        return ans
```



# 995

K连续位的最小翻转次数

关键：某一位置的元素值，跟其前面K-1个元素翻转的次数相关

滑动窗口：前K-1个元素中，哪些位置起始的子区 间进行了翻转

滑动窗口的元素个数就是当前位置已经被翻转的次数

如果被翻转了偶数次则不变，否则就变化；如果当前元素还要被翻转，就加到滑动窗口里

当i+K>n时，说明后面剩余的元素不够K个，无法翻转了，就失败了

```python
class Solution:
    def minKBitFlips(self, A: List[int], K: int) -> int:
        n=len(A)
        que=collections.deque()
        ans=0

        for i in range(n):
            if que and que[0]+K<=i:
                que.popleft()
            if len(que) % 2==A[i]:
                if i+K>n:
                    return -1
                que.append(i)
                ans+=1
        
        return ans
```



# 1004

最大连续1的个数

滑动窗口，rev表示窗口内0翻转为1的个数

```
class Solution:
    def longestOnes(self, A: List[int], K: int) -> int:
        left=0
        right=0

        ans=0
        rev=0
        while right<len(A):
            if A[right]==0:
                while rev>=K:
                    if A[left]==0:
                        rev-=1
                    left+=1
                rev+=1

            right+=1
            ans=max(ans,right-left)
        
        return ans
```



# 1438

绝对值不超过限制的最长连续子数组

滑动窗口+单调队列

用单调队列维护当前窗口的最大值和最小值

queMin单调增，queMax单调减

```python
class Solution:
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        n=len(nums)
        left=right=0
        ans=0
        queMax=deque()
        queMin=deque()

        while right<n:
            while queMax and queMax[-1]<nums[right]:
                queMax.pop()
            while queMin and queMin[-1]>nums[right]:
                queMin.pop()
            
            queMax.append(nums[right])
            queMin.append(nums[right])

            while queMin and queMax and queMax[0]-queMin[0]>limit:
                if nums[left]==queMin[0]:
                    queMin.popleft()
                if nums[left]==queMax[0]:
                    queMax.popleft()
                left+=1
            
            ans=max(ans,right-left+1)
            right+=1
        
        return ans
```



# 766 

托普利兹矩阵

按对角线遍历，比按行遍历更节约内存

注意矩阵不要越界

```c++
#include <algorithm>
class Solution {
public:
    bool isToeplitzMatrix(vector<vector<int>>& matrix) {
        int m=matrix.size();
        int n=matrix[0].size();
        if (m==1 || n==1)
            return true; 

        for(int i=0;i<n-1;i++){
            int e=matrix[0][i];
            int l=min(m,n-i);//不越界
            for(int j=1;j<l;j++){
                if(matrix[j][i+j]!=e){
                    return false;
                }
            }
        }

        for (int i=1;i<m-1;i++){
            int e=matrix[i][0];
            int l=min(n,m-i);//不越界
            for(int j=1;j<l;j++){
                if(matrix[i+j][j]!=e){
                    return false;
                }
            }
        }
        return true;
    }
};
```



# 169

多数元素

查找众数经典算法：摩尔投票法

c（候选者）和m（众数），当前的c遇到相同的，m就+1，否则m-1，m减到0时就换一个c

由于题目说一定存在，就不用再遍历一遍计数了

```c++
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int c=nums[0];
        int m=1;
        int n=nums.size();

        for (int i=1;i<n;i++){
            if (m==0){
                c=nums[i];
                m=1;
            }else{
                if (c==nums[i])
                    m++;
                else
                    m--;
            }
        }
        return c;
    }
};
```



# 206

反转链表

用两个指针遍历，因为要改next_p的next，所以要把next先存下来

```c++
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* p=head;
        if (!head)
            return nullptr;//当心链表为空
            
        ListNode* next_p=head->next;
        head->next=nullptr;

        while(next_p){
            ListNode* tmp=next_p->next;
            next_p->next=p;
            p=next_p;
            next_p=tmp;
        }

        return p;
    }
};
```

递归：head->next之后已经反向了，把head和head->next反向

```c++
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if (!head)
            return nullptr;
        if (!head->next)
            return head;

        ListNode* newhead=reverseList(head->next);
        head->next->next=head;
        head->next=nullptr;
        return newhead;
    }
};
```



# 226

翻转二叉树

递归

```c++
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if (!root) return nullptr;
        TreeNode* tmp=root->right;
        root->right=invertTree(root->left);
        root->left=invertTree(tmp);
        return root;
    }
};
```

遍历

```c++
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if (!root) return nullptr;
        queue<TreeNode*> que;
        que.push(root);

        while(!que.empty()){
            TreeNode* p=que.front();
            que.pop();
            TreeNode* tmp=p->left;
            p->left=p->right;
            p->right=tmp;
            if (p->left) 
                que.push(p->left);
            if (p->right) 
                que.push(p->right);
        }
        return root;
    }
};
```



# 234

回文链表

先用快慢指针找中点，慢指针遍历的同时把前一半链表反转

然后从中点向两端check

```c++
class Solution {
public:
    bool isPalindrome(ListNode* head) {
        if (!head || !head->next)
            return true;
        ListNode* slow=head;
        ListNode* slow_next=head->next;
        ListNode* fast=head;
        ListNode * tmp;

        while(fast->next && fast->next->next){
            fast=fast->next->next;
            //反转slow和slow->next
            tmp=slow_next->next;
            slow_next->next=slow;
            slow=slow_next;
            slow_next=tmp;
        }
        if (!fast->next)
            slow=slow->next;#节点数为奇数，跳过中点

        head->next=nullptr;
        while(slow_next){
            if(slow->val!=slow_next->val)
                return false;
            slow=slow->next;
            slow_next=slow_next->next;
        }
        
        return true;
    }
};
```



# 1052

爱生气的书店老板

先遍历一遍求初始值，再滑动窗口求最大

```c++
#include <algorithm>
class Solution {
public:
    int maxSatisfied(vector<int>& customers, vector<int>& grumpy, int X) {
        int n=customers.size();
        int ans=0;

        for(int i=0;i<n;i++){
            if (!grumpy[i]){
                ans+=customers[i];
            }
        }

        int now=ans;
        for(int i=0;i<X;i++){
            if (grumpy[i]){
                now+=customers[i];
            }
        }
        ans=max(ans,now);

        for(int i=X;i<n;i++){
            if (grumpy[i]){
                now+=customers[i];
            }
            if (grumpy[i-X]){
                now-=customers[i-X];
            }
            ans=max(ans,now);
        }
        return ans;
    }
};
```

 # 832

翻转图像

每一行内检查中心对称的元素，如果一样，就要反转，否则不用变化

如果一行的元素个数是奇数，中间的元素要单独反转

```c++
class Solution {
public:
    vector<vector<int>> flipAndInvertImage(vector<vector<int>>& A) {
        int m=A.size();
        int n=A[0].size();

        for(int i=0;i<m;i++){
            for(int j=0;j<n/2;j++){
                if(A[i][j]==A[i][n-1-j]){
                    A[i][j]=1-A[i][j];
                    A[i][n-1-j]=A[i][j];
                }
            }
            if (n%2){
                A[i][(n-1)/2]=1-A[i][(n-1)/2];
            }
        }

        return A;
    }
};
```



# 283

移动零

统计已经出现的零的个数，有几个零则非零元素就往前移动几位

零全部放在最后即可

```c++
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        int zero_num=0;
        int n=nums.size();
        for(int i=0;i<n;i++){
            if(!nums[i]){
                zero_num++;
            }else{
                nums[i-zero_num]=nums[i];
            }
        }
        for(int i=0;i<zero_num;i++){
            nums[n-1-i]=0;
        }
    }
};
```



 # 867

转置矩阵

easyyyyy

```c++
class Solution {
public:
    vector<vector<int>> transpose(vector<vector<int>>& matrix) {
        int m=matrix.size();
        int n=matrix[0].size();

        vector<vector<int>> trans(n, vector<int>(m));
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                trans[i][j]=matrix[j][i];
            }
        }
        return trans;
    }
};
```



# 461

汉明距离

方法一：把十进制转成二进制，每一位放在数组里

```c++
class Solution {
public:
    int hammingDistance(int x, int y) {
        int x_bin[32]={0};
        int y_bin[32]={0};
        dec2bin(x,x_bin);
        dec2bin(y,y_bin);

        int hamming=0;
        for (int i=0;i<32;i++){
            if (x_bin[i]!=y_bin[i]){
                hamming++;
            }
        }
        return hamming;
    }

    void dec2bin(int x,int bin[]){
        int i=31;

        while(x>0){
            bin[i]=x % 2;
            x=x>>1;
            i--;
        }
    }
};
```

方法二：先xor，再算二进制1的个数

```c++
class Solution {
public:
    int hammingDistance(int x, int y) {
        int xor_;
        xor_=x xor y;
        int hamming=0;

        while(xor_>0){
            hamming+=(xor_%2);
            xor_=xor_>>1;
        }

        return hamming;
    }
};
```

方法三：布莱恩$\cdot$ 克尼根算法

$x \& (x-1)$

每做一次可以移除最右边的1

减1之后，这个1变成0，右边的0全变成1，and一下这个1就没了

![img](https://pic.leetcode-cn.com/Figures/461/461_brian.png)

```c++
class Solution {
public:
    int hammingDistance(int x, int y) {
        int xor_=x xor y;
        int hamming=0;

        while(xor_){
            hamming+=1;
            xor_=xor_ & (xor_-1);
        }

        return hamming;
    }
};
```



# 543 

二叉树的直径

递归解法

对于某个节点，经过其的最长路径就是左右子树的最大深度之和+1

用递归求最大深度，顺便求最长路径

直径就是最长路径-1

```c++
class Solution {
public:
    int ans=1;
    int depth(TreeNode* root){
        if (!root){
            return 0;
        }
        int l=depth(root->left);
        int r=depth(root->right);
        ans=max(ans,l+r+1);
        return max(l,r)+1;
    }
    int diameterOfBinaryTree(TreeNode* root) {
        depth(root);
        return ans-1;
    }
};
```



# 617

合并二叉树

递归，改结点值和指针

```c++
class Solution {
public:
    TreeNode* mergeTrees(TreeNode* root1, TreeNode* root2) {
        if(!root1){
            return root2;
        }
        if(!root2){
            return root1;
        }

        root1->val+=root2->val;
        root1->left=mergeTrees(root1->left,root2->left);
        root1->right=mergeTrees(root1->right,root2->right);

        return root1;
    }
};
```



# 1178

猜字谜

二进制状态压缩+匹配

puzzle和word都用26位二进制来表示

word包含puzzle的第一个字母，且word的所有字母都在puzzle中

因此枚举puzzle后6位的每个子集，就得到每一个可能的word，再看word有没有出现

```c++
class Solution {
public:
    vector<int> findNumOfValidWords(vector<string>& words, vector<string>& puzzles) {
        unordered_map<int,int> frequency;

        //word转成二进制
        for(int i=0;i<words.size();i++){
            int mask=0;
            for(int j=0;j<words[i].length();j++){
                mask |= (1<<(words[i][j]-'a'));//对应位上为1
            }
            frequency[mask]++;
        }

        vector<int> ans;
        for(int i=0;i<puzzles.size();i++){
            int total=0;
			
            //枚举子集
            for(int choose=0;choose<(1<<6);choose++){
                int mask=1<<(puzzles[i][0]-'a');//第一个字母一定有
                for(int j=0;j<6;j++){
                    if(choose & (1<<j)){
                        mask |= (1<<(puzzles[i][j+1]-'a'));
                    }
                }
                total+=frequency[mask];
            }
            ans.push_back(total);
        }
        return ans;
    }
};
```



# 395

至少有K个重复字符的最长字串

方法一：分治

如果存在某个字符ch，其出现次数大于0小于k，则任何包含ch的子串都不可能满足要求

用ch将字符串切成若干段，满足要求的最长字串一定出现在某个被切分的段内，而不能跨越

对于这些段用分治的方法进行递归

时间复杂度：$O(N\cdot |\Sigma|)$ $N$为字符串长度， $\Sigma$ 为字符集大小，本题中为26。由于每次递归都会**完全去除**某个字符，因此递归深度最多为$|\Sigma|$ 

空间复杂度：$O(|\Sigma|^2)$ 每次递归都需要$O(|\Sigma|)$的额外空间

```c++
class Solution {
public:
    int dfs(string s, int l, int r, int k){
        //统计每个字符出现的次数
        int count[26]={0};
        for(int i=l;i<r;i++){
            count[s[i]-'a']++;
        }

        //找一个ch
        char spilt=0;
        for(int i=0;i<26;i++){
            if(count[i] && count[i]<k){
                spilt=i+'a';
            }
        }
        //没找到则说明这段字串满足要求
        if(!spilt){
            return r-l;
        }

        int i=l;
        int ret=0;
        while(i<r){
            while(i<r && s[i]==spilt){
                i++;
            }
            if (i>=r){
                break;
            }
            int start=i;
            while(i<r && s[i]!=spilt){
                i++;
            }
            ret=max(ret,dfs(s,start,i,k));//递归
        }
        return ret;
    }

    int longestSubstring(string s, int k) {
        return dfs(s,0,s.length(),k);
    }
};
```

方法二：滑动窗口

枚举最长字串中的字符种类数目，最小为1，最大为$|\Sigma|$ 

对于给定的字符种类数目t，维护滑动窗口，使其中字符种类数目tot不多于t

然后判断滑动窗口是否满足要求：维护一个计数器less，代表当前出现次数小于k的字符的数量，无需遍历子串

```c++
class Solution {
public:
    int longestSubstring(string s, int k) {
        int ret=0;
        int n=s.length();
        for (int t=1;t<=26;t++){
            int l=0,r=0;
            int count[26]={0};
            int tot=0,less=0;

            while(r<n){
                //右边界右移
                count[s[r]-'a']++;
                if (count[s[r]-'a']==1){
                    tot++;
                    less++;
                }
                if (count[s[r]-'a']==k){
                    less--;
                }

                while(tot>t){
                    //左边界右移
                    count[s[l]-'a']--;
                    if(count[s[l]-'a']==k-1){
                        less++;
                    }
                    if(!count[s[l]-'a']){
                        tot--;
                        less--;
                    }
                    l++;
                }

                if (!less){
                    ret=max(ret,r-l+1);
                }
                r++;
            }
        }
        return ret;
    }
};
```



# 896

单调数列

方法一：两次遍历，分别判断单调增和单调减

```c++
class Solution {
public:
    bool isMonotonic(vector<int>& A) {
        bool flag=true;
        for(int i=0;i<A.size()-1;i++){
            if (A[i]>A[i+1]){
                flag=false;
                break;
            }
        }
        if (flag) return true;
        flag=true;
        for(int i=0;i<A.size()-1;i++){
            if (A[i]<A[i+1]){
                flag=false;
                break;
            }
        }
        return flag;
    }
};
```

方法二：

如果数列里同时有严格单调增和严格单调减，则不是单调数列

```c++
class Solution {
public:
    bool isMonotonic(vector<int>& A) {
        bool inc=false,dec=false;
        for(int i=0;i<A.size()-1;i++){
            if (A[i]<A[i+1]){
                inc=true;
            }else if(A[i]>A[i+1]){
                dec=true;
            }
            if (inc and dec){
                return false;
            }
        }
        return true;
    }
};
```



# 2

两数相加

可以认为长度较短的链表后面都是0

如果最后有进位，还会多一个节点

```c++
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode *p1=l1,*p2=l2,*head=nullptr,*tail=nullptr;
        int carry=0;
        while(p1 || p2){
            int val1=p1? p1->val:0;
            int val2=p2? p2->val:0;
            int sum=val1+val2+carry;
            if(sum>=10){
                carry=1;
                sum=sum % 10;
            }else{
                carry=0;
            }
            if(!head){
                head=tail=new ListNode(sum);
            }else{
                tail->next=new ListNode(sum);
                tail=tail->next;
            }
            if(p1)
                p1=p1->next;
            if(p2)
                p2=p2->next;
        }
        
        if(carry){
            tail->next=new ListNode(1);
        }
        return head;
    }
};
```



# 303

区域和检索

前缀和，dp[i]是0~i-1元素的和

```c++
class NumArray {
private:
    int* dp;
public:
    NumArray(vector<int>& nums) {
        int n=nums.size();
        dp=new int[n+1];

        dp[0]=0;
        for(int i=1;i<=n;i++){
            dp[i]=dp[i-1]+nums[i-1];
        }
    }
    
    int sumRange(int i, int j) {
        return dp[j+1]-dp[i];
    }
};
```



# 3

无重复字符的最长字串

滑动窗口，由于任何字符都能出现在字符串中，可以用ascii码

```c++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        bool freq[128]={0};
        int left=0,right=0;
        int ans=0;

        while(right<s.length()){
            if(freq[s[right]]){
                ans=max(ans,right-left);
                while(s[left]!=s[right]){
                    freq[s[left]]=false;
                    left++;
                }
                left++;
            }
            freq[s[right]]=true;
            right++;
        }
        ans=max(ans,right-left);
        return ans;
    }
};
```



# 304

二维区域和检索-矩阵不可变

二维前缀和

```c++
class NumMatrix {
private:
    vector<vector<int>> dp;
public:
    NumMatrix(vector<vector<int>>& matrix) {
        int m=matrix.size();
        if(!m) return;
        int n=matrix[0].size();

        dp.resize(m+1,vector<int>(n+1));
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                dp[i+1][j+1]=dp[i+1][j]+dp[i][j+1]-dp[i][j]+matrix[i][j];
            }
        }
    }
    
    int sumRegion(int row1, int col1, int row2, int col2) {
        return dp[row2+1][col2+1]+dp[row1][col1]-dp[row1][col2+1]-dp[row2+1][col1];
    }
};
```



# 338

比特位计数

方法一：去掉最高位的1

用```now_max```保存现在最大的2的倍数，其只有一个1，可以用```i&(i-1)```是否为0来判断

对于其它的数，只比```i-now_max```多最高位的1

```c++
class Solution {
public:
    vector<int> countBits(int num) {
        vector<int> ans(num+1);
        
        int now_max=0;
        for(int i=1;i<=num;i++){
            if((i&(i-1))==0){
                now_max=i;
                ans[i]=1;
            }else{
                ans[i]=ans[i-now_max]+1;
            }
        }
        return ans;
    }
};
```

方法二：去掉最右边的1

```i&(i-1)```可以把```i```最右边的1去掉

```c++
class Solution {
public:
    vector<int> countBits(int num) {
        vector<int> ans(num+1);
        
        for(int i=1;i<=num;i++){
            ans[i]=ans[i&(i-1)]+1;
        }
        return ans;
    }
};
```



# 5

最长回文子串

方法一：dp

从长度为1的子串开始dp：长度为1肯定是回文，长度为2就看两个字符相不相等

[i,j]的子串就看[i+1,j-1]和两端的字符

```c++
class Solution {
public:
    string longestPalindrome(string s) {
        int n=s.length();
        bool dp[n][n];

        string ans="";
        for(int l=0;l<n;l++){
            for(int i=0;i+l<n;i++){
                if(!l){
                    dp[i][i]=true;
                }else if(l==1){
                    dp[i][i+1]=(s[i]==s[i+1]);
                }else{
                    dp[i][i+l]=dp[i+1][i+l-1] && (s[i]==s[i+l]);
                }
                if(dp[i][i+l] && l+1>ans.length()){
                    ans=s.substr(i,l+1);
                }
            }
        }
        return ans;
    }
};
```

方法二：中心拓展

dp的边界情况就是长度为1或2的子串，所以从这些中心出发，向两侧拓展

```c++
class Solution {
public:
    string longestPalindrome(string s) {
        int start=0,end=0;
        for(int i=0;i<s.length();i++){
            int len=max(CenterExpand(s,i,i),CenterExpand(s,i,i+1));
            if(len>end-start+1){
                start=i-(len-1)/2;//len要减1，否则长度为偶数时会多一个
                end=i+len/2;
            }
        }
        return s.substr(start,end-start+1);
    }

    int CenterExpand(string s, int left, int right){
        while(left>=0 && right<=s.length() && s[left]==s[right]){
            left--;
            right++;
        }
        return right-left-1;//不包括right和left
    }
};
```



# 11

盛最多水的容器

双指针：从两边向中间移动

当前的容量=两个指针指向数字的较小值*指针之间的距离

每次移动时，应该移动指向数字较小的那个，这样前者才会增大

```c++
class Solution {
public:
    int maxArea(vector<int>& height) {
        int ans=0;
        int left=0;
        int right=height.size()-1;

        while(left<right){
            int area=(right-left)*min(height[left],height[right]);
            ans=max(ans,area);
            if(height[left]<height[right]){
                left++;
            }else{
                right--;
            }
        }
        return ans;
    }
};
```



# 354

俄罗斯套娃信封问题

先在宽度上进行升序排序，然后根据高度选一个最长严格递增子序列

对于宽度一样的情况，按照高度降序排序，以保证不会出现同样宽度套娃的情况

计算最长严格递增子序列：

方法一：动态规划

对于第i个元素，找能排在其前面的元素j，```f[i]=max(f[i],f[j]+1)```

如果找不到j，就是i本身

```max_element```返回数组里最大的元素的迭代器（指针）

```c++
class Solution {
public:
    int maxEnvelopes(vector<vector<int>>& envelopes) {
        int n=envelopes.size();
        
        sort(envelopes.begin(),envelopes.end(),compare);
        vector<int> f(n,1);

        for(int i=0;i<n;i++){
            for (int j=0;j<i;j++){
                if(envelopes[j][1]<envelopes[i][1]){
                    f[i]=max(f[i],f[j]+1);
                }
            }
        }
        return *max_element(f.begin(),f.end());

    }

    static bool compare(vector<int> &e1, vector<int> &e2){
        return e1[0]<e2[0] || (e1[0]==e2[0] && e1[1]>e2[1]);
    }
};
```

方法二：二分查找

f[j]维护当前可以组成的长度为j的严格递增子序列的末尾元素的最小值，f肯定是严格单增的

对于下一个元素i：

如果i比f的最后一个值f[j]大，则i可以接在后面，形成长度为j+1的子序列；

否则找出比i严格小的最大元素，把i接在其后，因为i肯定比原来这个位置上的元素要小，更有可能找到更长的子序列

```lower_bound```可以找到数组里大于等于```num```的最小下标（前提是数组是升序的）

```
class Solution {
public:
    int maxEnvelopes(vector<vector<int>>& envelopes) {
        int n=envelopes.size();
        
        sort(envelopes.begin(),envelopes.end(),compare);
        vector<int> f={envelopes[0][1]};

        for(int i=1;i<n;i++){
            int num=envelopes[i][1];
            if(num>f.back()){
                f.push_back(num);
            }else{
                auto it=lower_bound(f.begin(),f.end(),num);
                *it=num;
            }
        }
        return f.size();
    }

    static bool compare(vector<int> &e1, vector<int> &e2){
        return e1[0]<e2[0] || (e1[0]==e2[0] && e1[1]>e2[1]);
    }
};
```



# 232

用栈实现队列

用一个输入栈一个输出栈，每次要取队列头元素的时候，先看输出栈里有没有，没有就把输入栈的先pop出来，再push进输出栈，这样顺序就是FIFO的

```c++
class MyQueue {
private:
    stack<int> in_stack;
    stack<int> out_stack;
public:
    /** Initialize your data structure here. */
    MyQueue() {

    }
    
    /** Push element x to the back of queue. */
    void push(int x) {
        in_stack.push(x);
    }
    
    /** Removes the element from in front of queue and returns that element. */
    int pop() {
        int r;
        if(!out_stack.empty()){
            r=out_stack.top();
            out_stack.pop();
        }else{
            while(in_stack.size()!=1){
                out_stack.push(in_stack.top());
                in_stack.pop();
            }
            r=in_stack.top();
            in_stack.pop();
        }
        return r;
    }
    
    /** Get the front element. */
    int peek() {
        int r;
        if(!out_stack.empty()){
            r=out_stack.top();
        }else{
            while(in_stack.size()){
                out_stack.push(in_stack.top());
                in_stack.pop();
            }
            r=out_stack.top();
        }
        return r;
    }
    
    /** Returns whether the queue is empty. */
    bool empty() {
        return in_stack.empty() && out_stack.empty();
    }
};
```



# 503

下一个更大元素II

单调栈（单调递减），保存下标

移动到位置$i$时，就将栈中所有对应值小于nums[i]的下标弹出，因为这些值的下一个更大元素就是nums[i]

由于是循环队列，需要再遍历回来，即把前n-1个元素拼接在原数组后面

实现时不用显式地拼接，对下标取模即可

```c++
class Solution {
public:
    vector<int> nextGreaterElements(vector<int>& nums) {
        int n=nums.size();
        vector<int> ans(n,-1);
        stack<int> s;

        for(int i=0;i<2*n-1;i++){
            while(!s.empty() && nums[s.top()]<nums[i%n]){
                ans[s.top()%n]=nums[i%n];
                s.pop();
            }
            s.push(i%n);
        }
        return ans;
    }
};
```



# 15

三数之和

排序+双指针

为了避免有重复的，先对数组进行排序，然后保证三元组是$a\le b\le c$

在每一轮循环内，相邻两次枚举的元素不能相同

固定了前两重循环的元素后，第三个元素可以从后往前找，这样第二重和第三重循环可以简化成双指针

时间复杂度为$O(N^2)$

```c++
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        sort(nums.begin(),nums.end());
        int n=nums.size();
        vector<vector<int>> ans;

        for(int i=0;i<n;i++){//第一个元素
            if (!i || nums[i]!=nums[i-1]){//不重复枚举
                int k=n-1;//第三个元素
                int target=-nums[i];
                for(int j=i+1;j<n;j++){//第二个元素
                    if (j==i+1 || nums[j]!=nums[j-1]){//不重复枚举
                        while(j<k && nums[j]+nums[k]>target){//双指针移动
                            k--;
                        }
                        if(j>=k){//第二个元素要小于第三个元素
                            break;
                        }
                        if(nums[j]+nums[k]==target){
                            ans.push_back({nums[i],nums[j],nums[k]});
                        }
                    }
                }
            }
        }
        return ans;
    }
};
```



# 131

分割回文串

方法一：顺序遍历尝试划分，先找到一个回文串，然后递归分割剩余的字符串

非常慢且空间占用多

```c++
class Solution {
public:
    vector<vector<string>> partition(string s) {
        vector<vector<string>> ans;
        if(s.length()==1){
            vector<string> tmp;
            tmp.push_back(s);
            ans.push_back(tmp);
            return ans;
        }

        for (int i=0;i<s.length();i++){
            if(!i || check(s.substr(0,i+1))){//找到回文串
                if(i+1!=s.length()){
                    vector<vector<string>> ret=partition(s.substr(i+1));//递归划分剩余的
                    for(int j=0;j<ret.size();j++){
                        ret[j].insert(ret[j].begin(),s.substr(0,i+1));//后面划分好的，加上前面的
                    }
                    ans.insert(ans.end(),ret.begin(),ret.end());
                }else{//整个串都是回文
                    vector<string> tmp;
                    tmp.push_back(s);
                    ans.push_back(tmp);
                }
            }
        }
        return ans;
    }

    bool check(string s){//检查是不是回文
        int n=s.length();
        for(int i=0;i<n/2;i++){
            if(s[i]!=s[n-1-i]){
                return false;
            }
        }
        return true;
    }
};
```

方法二：回溯+dp

用dp来判断所有子串是否为回文，i>=j的dp\[i][j]都是true，这样可以不用管单个字符，也不用管字符长度够不够dp\[j+1][j+i-1]

然后用dfs来分割

时间复杂度：$O(n\cdot2^n)$ ，n为字符串的长度。最坏情况下，字符串包含n个相同的字符，划分方案数为$2^{n-1}=O(2^n)$，每一种划分需要$O(n)$的时间。dp所需的时间是$O(n^2)$，比$O(n\cdot2^n)$小，可以忽略。

空间复杂度：$O(n^2)$，不考虑答案所需要的空间，dp所需的空间为$O(n^2)$，回溯中需要$O(n)$的空间

```c++
class Solution {
private:
    int n;
    vector<string> ans;
    vector<vector<string>> ret;
    vector<vector<bool>> dp;
public:
    vector<vector<string>> partition(string s) {
        n=s.length();
        dp.resize(n,vector<bool>(n,true));

        for(int i=1;i<n;i++){
            for(int j=0;j+i<n;j++){
                dp[j][j+i]=(s[j]==s[j+i]) && dp[j+1][j+i-1];
            }
        }

        dfs(s,0);
        
        return ret;
    }

    void dfs(string &s,int i){
        if(i==n){
            ret.push_back(ans);//找到一个分割
            return;
        }

        for(int j=i;j<n;j++){
            if(dp[i][j]){
                ans.push_back(s.substr(i,j-i+1));
                dfs(s,j+1);
                ans.pop_back();//回溯
            }
        }
    }
};
```



# 17

电话号码的字母组合

dfs回溯

```c++
class Solution {
private:
    int n;
    vector<string> ans;
    string t="";

public:
    vector<string> letterCombinations(string digits) {
        n=digits.size();
        if(!n){
            return ans;
        }

        dfs(digits,0);
        return ans;
    }

    void dfs(const string &s, int i){
        if(i==n){
            ans.push_back(t);
            return;
        }
        if(s[i]<'7'){
            for(int j=0;j<3;j++){
                char c=3*(s[i]-'1')-3+j+'a';
                t+=c;
                dfs(s,i+1);
                t=t.substr(0,i);
            }
        }else if(s[i]=='7'){
            for(char c='p';c<='s';c++){
                t+=c;
                dfs(s,i+1);
                t=t.substr(0,i);
            }
        }else if(s[i]=='8'){
            for(char c='t';c<='v';c++){
                t+=c;
                dfs(s,i+1);
                t=t.substr(0,i);
            }
        }else{
            for(char c='w';c<='z';c++){
                t+=c;
                dfs(s,i+1);
                t=t.substr(0,i);
            }
        }
    }
};
```

优化：用哈希表把数字和字母的对应关系存下来，代码简单一些

and String支持线性表的操作

```c++
class Solution {
private:
    int n;
    vector<string> ans;
    string t="";
    unordered_map<char,string> num2char{
            {'2', "abc"},
            {'3', "def"},
            {'4', "ghi"},
            {'5', "jkl"},
            {'6', "mno"},
            {'7', "pqrs"},
            {'8', "tuv"},
            {'9', "wxyz"}
        };

public:
    vector<string> letterCombinations(string digits) {
        n=digits.size();
        if(!n){
            return ans;
        }

        dfs(digits,0);
        return ans;
    }

    void dfs(const string &s, int i){
        if(i==n){
            ans.push_back(t);
            return;
        }
        const string &m=num2char[s[i]];
        for(int j=0;j<m.length();j++){
            t.push_back(m[j]);
            dfs(s,i+1);
            t.pop_back();
        }
    }
};
```



# 132

分割回文串II

dp：设$f[i]$表示$s[0\dots i]$的最少分割次数，可以枚举分割出的最后一个回文串

$$f[i]=\min_{0\le j\le i}\{f[j]\}+1$$，其中$s[j+1\dots i]$是回文串

还要考虑$s[0\dots i]$本身是回文串的情况，即$f[i]=0$

判断回文串可以用**131**中预处理方法

```c++
class Solution {
public:
    int minCut(string s) {
        int n=s.length();
        vector<vector<bool>> dp(n,vector<bool>(n,true));

        for(int i=1;i<n;i++){
            for(int j=0;j+i<n;j++){
                dp[j][j+i]=(s[j]==s[j+i]) && dp[j+1][j+i-1];
            }
        }

        vector<int> div(n);

        for(int i=0;i<n;i++){
            if(dp[0][i]){
                div[i]=0;
            }else{
                int t=i;
                for(int j=1;j<=i;j++){
                    if(dp[j][i]){
                        t=min(t,div[j-1]);
                    }
                }
                div[i]=t+1;
            }
        } 
        return div[n-1];
    }
};
```



# 19

删除链表的倒数第N个节点

方法一：计算链表长度，需要两次遍历

方法二：栈，用栈把所有节点存进去，然后弹出的第n个就是要删除的，栈顶元素就是其前驱节点

方法三：双指针

用快慢两个指针遍历链表，快指针比慢指针超前n个节点，当快指针到达链表末尾时，慢指针就位于倒数第n个节点

找被删除节点的前驱节点对删除操作更加方便，因此在头节点前增加一个哑节点，让慢指针从哑节点开始

最终慢指针的下一个节点就是要被删除的节点

```c++
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode* prev=new ListNode(0,head);
        ListNode* slow=prev;
        ListNode* fast=head;

        for(int i=0;i<n;i++){
            fast=fast->next;
        }

        while(fast){
            fast=fast->next;
            slow=slow->next;
        }
        ListNode* tmp=slow->next;
        slow->next=tmp->next;
        delete tmp;

        return prev->next;
    }
};
```



# 1047

删除字符串中的所有相邻重复项

方法一：傻瓜方法

双指针找相邻的两个字符

用```del```表示字符有没有被删掉

如果一次遍历中有删掉的，就再遍历一次

```c++
class Solution {
public:
    string removeDuplicates(string S) {
        int n=S.length();
        vector<bool> del(n,false);

        bool flag;
        do{
            flag=false;
            int i=0,j=0;
            while(i<n-1){
                while(del[i]) i++;//找没被删掉的i
                j=i+1;
                while(del[j]) j++;//找没被删掉的j

                if(S[i]==S[j]){
                    del[i]=del[j]=true;
                    flag=true;
                    i=j+1;//i移动到j下一个
                }else{
                    i=j;//i移动到j
                }
            }
        }while(flag);

        string ans;
        for(int i=0;i<n;i++){
            if(!del[i]){
                ans+=S[i];
            }
        }
        return ans;
    }
};
```

高级方法：用栈

每来一个元素，就跟栈顶元素比较，如果一样，就把栈顶元素pop出来，否则入栈

```c++
class Solution {
public:
    string removeDuplicates(string S) {
        string ans;

        for(int i=0;i<S.length();i++){
            if(ans.empty() || ans.back()!=S[i]){
                ans.push_back(S[i]);
            }else{
                ans.pop_back();
            }
        }
        return ans;
    }
};
```



# 1766

互质树

超时方法：dfs遍历+模拟

```c++
class Solution {
private:
    vector<vector<int>> prime;
public:
    vector<int> getCoprimes(vector<int>& nums, vector<vector<int>>& edges) {
        int n = nums.size();
        vector<bool> visited(n-1, false);
        prime.resize(51, vector<int>(51, 0));

        vector<int> ans(n, -1);
        vector<int> q;

        q.push_back(0);

        while (!q.empty()) {
            int node = q.back();

            for (int i = q.size() - 2; i >= 0; i--) {
                if (gcd(nums[q[i]], nums[node])) {
                    ans[node] = q[i];
                    break;
                }
            }

            int child=-1;
            for (int i = 0; i < n-1; i++) {
                if (visited[i]) {
                    continue;
                }
                if (edges[i][0] == node) {
                    child = edges[i][1];
                }
                if (edges[i][1] == node) {
                    child = edges[i][0];
                }
                if(child!=-1){
                    visited[i]=true;
                    break;
                }
            }

            if (child != -1) {
                q.push_back(child);
            }else{
                q.pop_back();
            }
        }

        return ans;
    }

    bool gcd(int a, int b) {
        if (a < b) {
            swap(a, b);
        }
        if (prime[a][b]) {
            return prime[a][b] == 1;
        }
        if (a % 2 == 0 && b % 2 == 0) {
            prime[a][b] = -1;
        } else {
            int a1=a,b1=b;
            int r;
            while (a1 % b1) {
                r = a1 % b1;
                a1 = b1;
                b1 = r;
            }
            prime[a][b] = (b1 > 1 ? -1 : 1);
        }
        return prime[a][b] == 1;
    }
};
```

好方法：栈+节点值

如果蛮力检查一个节点的祖先节点，一个节点的祖先节点最多会有n-1个，因此会超时

换一种思路，因为$nums[i]\le 50$，可以从节点值出发，枚举与节点值互素的数，并对每个数找出离节点最近的点

对于任一数字，找最近的祖先节点：dfs，1-50每一个值维护一个栈，然后把节点放到值对应的栈里，并用节点的深度来判断距离

```c++
class Solution {
public:
    vector<vector<int>> G;
    stack<pair<int,int>> lasts[51];
    vector<int> ans;
    void dfs(int node, int pre, int level, vector<int>& nums) {
        int fa = -1, depth = -1;
        for(int i = 1; i <= 50; ++i) {
            if(lasts[i].size() && lasts[i].top().first > depth && __gcd(i, nums[node]) == 1) {
                fa = lasts[i].top().second;
                depth = lasts[i].top().first;
            }
        }
        ans[node] = fa;
        for(int ne : G[node]) {
            if(ne != pre) {
                lasts[nums[node]].push({level, node});
                dfs(ne, node, level + 1, nums);
                lasts[nums[node]].pop();
            }
        }
    }
    vector<int> getCoprimes(vector<int>& nums, vector<vector<int>>& edges) {
        int n = nums.size();
        G.resize(n);//构造邻接矩阵
        for(auto& e : edges) {
            G[e[0]].push_back(e[1]);
            G[e[1]].push_back(e[0]);
        }
        ans.resize(n);
        dfs(0, -1, 0, nums);
        return ans;
    }
};
```



# 224

基本计算器

方法一：经典符号栈

0：+，1：-，2：（

```c++
class Solution {
public:
    stack<int> dataStack;
    stack<int> opStack;
    int calculate(string s) {
        int prev=0;

        for(int i=0;i<s.length();i++){
            char ch=s[i];
            switch(ch){
                case ' ':
                    continue;
                case '+':case '-':
                    while(!opStack.empty() && opStack.top()!=2){
                        OP(opStack.top());//左括号前的都可以计算
                        opStack.pop();
                    }
                    //新符号进栈
                    if(ch=='+'){
                        opStack.push(0);
                    }else{
                        opStack.push(1);
                    }
                    break;
                case '(':
                    opStack.push(2);//左括号进栈
                    break;
                case ')':
                    while(!opStack.empty() && opStack.top()!=2){//左右括号里的都可以计算
                        OP(opStack.top());
                        opStack.pop();
                    }
					opStack.pop();//左括号出栈
                    break;
                default:
                    int num=ch-'0';
                    prev=prev*10+num;//连续数字的计算
                    if(s[i+1]<'0' || s[i+1]>'9'){
                        dataStack.push(prev);//数字结束了
                        prev=0;//重置
                    }
                    
            }
        }

        while(!opStack.empty()){
            OP(opStack.top());
            opStack.pop();
        }
        return dataStack.top();
    }

    void OP(int op){//计算表达式
        int num1=0,num2;

        num2=dataStack.top();
        dataStack.pop();

        if(!dataStack.empty()){//有负数的情况，可以看作0-num2
            num1=dataStack.top();
        dataStack.pop();
        }
        
        if(!op){
            dataStack.push(num1+num2);
        }else{
            dataStack.push(num1-num2);
        }
    }
};
```

方法二：括号展开+栈

因为只有加法和减法两种运算符，可以将表达式中的括号展开

如果遇到左括号，就把括号前的符号用栈记录下来，+1表示+，-1表示-

对于括号内的符号，如果括号前（栈顶）的符号是-，那么当前符号就要翻转

```c++
class Solution {
public:
    int calculate(string s) {
        int prev=0,ans=0;
        stack<int> opStack;
        int sign=1;
        opStack.push(1);//可以看作整个表达式括号括起来，外面是+号

        for(int i=0;i<s.length();i++){
            char ch=s[i];
            switch(ch){
                case ' ':
                    continue;
                case '+':
                    sign=opStack.top();
                    break;
                case '-':
                    sign=-opStack.top();
                    break;
                case '(':
                    opStack.push(sign);
                    break;
                case ')':
                    opStack.pop();
                    break;
                default:
                    int num=ch-'0';
                    prev=prev*10+num;
                    if(s[i+1]<'0' || s[i+1]>'9'){
                        ans+=sign*prev;//数字结束，将值加到结果中
                        prev=0;
                    }
            }
        }
        return ans;
    }
};
```



# 22

括号生成

dfs回溯，left和right分别表示剩余的左右括号数量

```c++
class Solution {
public:
    vector<string> ans;
    string s; 
    vector<string> generateParenthesis(int n) {
        s.push_back('(');
        dfs(n-1,n);

        return ans;
    }
    void dfs(int left,int right){
        if(!left && !right){
            ans.push_back(s);
            return;
        }
        if(left>0){//可以加一个左括号
            s.push_back('(');
            dfs(left-1,right);
            s.pop_back();
        }
        if(right>0 && left<right){//可以加一个右括号
            s.push_back(')');
            dfs(left,right-1);
            s.pop_back();
        }
    }
};
```



# 227

基本计算器II

方法一：符号栈

0：+，1：-，2：（，3：*，4：/

```c++
class Solution {
public:
    stack<int> dataStack;
    stack<int> opStack;
    int calculate(string s) {
        int prev=0;

        for(int i=0;i<s.length();i++){
            char ch=s[i];
            switch(ch){
                case ' ':
                    continue;
                case '+':case '-':
                    while(!opStack.empty() && opStack.top()!=2){
                        OP(opStack.top());//左括号前的都可以计算
                        opStack.pop();
                    }
                    //新符号进栈
                    if(ch=='+'){
                        opStack.push(0);
                    }else{
                        opStack.push(1);
                    }
                    break;
                case '*':case '/':
                    while(!opStack.empty() && opStack.top()>=3){
                        OP(opStack.top());
                        opStack.pop();
                    }
                    if(ch=='*'){
                        opStack.push(3);
                    }else{
                        opStack.push(4);
                    }
                    break;
                case '(':
                    opStack.push(2);//左括号进栈
                    break;
                case ')':
                    while(!opStack.empty() && opStack.top()!=2){//左右括号里的都可以计算
                        OP(opStack.top());
                        opStack.pop();
                    }
					opStack.pop();//左括号出栈
                    break;
                default:
                    int num=ch-'0';
                    prev=prev*10+num;//连续数字的计算
                    if(s[i+1]<'0' || s[i+1]>'9'){
                        dataStack.push(prev);//数字结束了
                        prev=0;//重置
                    }
                    
            }
        }

        while(!opStack.empty()){
            OP(opStack.top());
            opStack.pop();
        }
        return dataStack.top();
    }

    void OP(int op){//计算表达式
        int num1=0,num2;

        num2=dataStack.top();
        dataStack.pop();

        if(!dataStack.empty()){//有负数的情况，可以看作0-num2
            num1=dataStack.top();
        dataStack.pop();
        }
        
        switch(op){
            case 0:
                dataStack.push(num1+num2);break;
            case 1:
                dataStack.push(num1-num2);break;
            case 3:
                dataStack.push(num1*num2);break;
            case 4:
                dataStack.push(num1/num2);
        }
    }
};
```

方法二：因为没有括号，所以可以即时算出乘除的结果

当扫描到符号或结尾时，根据上一个符号，将新的数字入栈或把栈顶数字乘除新的数字

除了最后的空格，其它都无视

最后把栈里的数字求和

````c++
class Solution {
public:
    int calculate(string s) {
        vector<int> stack;
        char presign='+';

        int prev=0;
        for(int i=0;i<s.length();i++){
            char ch=s[i];
            if(ch>='0' && ch<='9'){
                int num=ch-'0';
                prev=prev*10+num;
            }

            if(((ch<'0' || ch>'9') && ch!=' ') || i==s.length()-1){
                switch(presign){
                    case '+':
                        stack.push_back(prev);
                        break;
                    case '-':
                        stack.push_back(-prev);
                        break;
                    case '*':
                        stack.back()*=prev;
                        break;
                    case '/':
                        stack.back()/=prev;            
                }
                presign=ch;
                prev=0;
            }
        }
        return accumulate(stack.begin(),stack.end(),0);
    }
};
````



# 31

下一个排列

C++标准库函数写法

原理：

我们需要将一个左边的「较小数」与一个右边的「较大数」交换，以能够让当前排列变大，从而得到下一个排列

同时我们要让这个「较小数」尽量靠右，而「较大数」尽可能小。当交换完成后，「较大数」右边的数需要按照升序重新排列。这样可以在保证新排列大于原来排列的情况下，使变大的幅度尽可能小

做法：

首先从后向前查找第一个顺序对 (i,i+1)，满足 $a[i] < a[i+1]$。这样「较小数」即为 $a[i]$。此时 $[i+1,n)$必然是下降序列

如果找到了顺序对，那么在区间$[i+1,n)$中从后向前查找第一个元素 j满足 $a[i] < a[j]$。这样「较大数」即为 $a[j]$

交换 $a[i]$ 与 $a[j]$，此时可以证明区间 $[i+1,n)$ 必为降序。我们可以直接使用双指针反转区间 $[i+1,n)$ 使其变为升序，而无需对该区间进行排序

如果较小数不存在，则说明整个排列是降序的，直接反转即可

```c++
class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        int n=nums.size();
        int i=n-2,j;
        while(i>=0 && nums[i]>=nums[i+1]) i--;//找较小数

        if(i>=0){//较小数存在
            j=n-1;
            while(nums[i]>=nums[j]) j--;//找较大数
            swap(nums[i],nums[j]);
        }

        //反转
        i++;
        j=n-1;
        while(i<j){
            swap(nums[i],nums[j]);
            i++;
            j--;
        }
        //reverse(nums.begin()+i+1,nums.end());
    }
};
```



# 331

验证二叉树的前序序列化

对于一个正确的二叉树，入度之和等于出度之和

空节点：0个出度，1个入度

非空节点：2个出度，1个入度

遍历到任意节点时，出度应该大于等于入度，原因是还没遍历到该节点的子节点

遍历完成后，整棵树的出度应该等于入度

diff=出度-入度，初始化为1的原因是根节点虽然是非空节点，但其入度为0，抵消其减去的1个入度

```c++
class Solution {
public:
    bool isValidSerialization(string preorder) {
        int diff=1;
        int i=0;
        while(i<preorder.length()){
            if(preorder[i]!=','){
                while(i<preorder.length()-1 && preorder[i+1]!=','){
                    i++;
                }
                diff-=1;
                if(diff<0){
                    return false;
                }
                if(preorder[i]!='#'){
                    diff+=2;
                }
            }
            i++;
        }
        return diff==0;
    }
};
```



# 705

设计哈希集合

哈希表：开散链表的实现

```c++
class MyHashSet {
private:
    struct node{
        int data;
        node* next;

        node(const int d,node *n=nullptr){
            data=d;
            next=n;
        }
        node(){next=nullptr;}
    };

    node **arr;
    int size=1001;
public:
    /** Initialize your data structure here. */
    MyHashSet() {
        arr=new node*[size];
        for(int i=0;i<size;i++){
            arr[i]=nullptr;
        }
    }
    
    void add(int key) {
        int pos=key%size;
        node *p=arr[pos];
        while(p && p->data!=key) p=p->next;
        if(!p){//不重复添加
            arr[pos]=new node(key,arr[pos]);
        }
    }
    
    void remove(int key) {
        int pos=key%size;
        if(!arr[pos]) return;

        node *p=arr[pos];
        if(arr[pos]->data==key){
            arr[pos]=p->next;
            delete p;
            return;
        }

        while(p->next && p->next->data!=key) p=p->next;
        if(p->next){
            node *q=p->next;
            p->next=q->next;
            delete q;
        }
    }
    
    /** Returns true if this set contains the specified element */
    bool contains(int key) {
        int pos=key%size;
        node *p=arr[pos];
        while(p && p->data!=key) p=p->next;
        return p!=nullptr;
    }
};
```

STL容器写法

```c++
class MyHashSet {
private:
    vector<list<int>> arr;
    static const int size=1001;
public:
    /** Initialize your data structure here. */
    MyHashSet() {
        arr.resize(size);
    }
    
    void add(int key) {
        int pos=key%size;
        for(auto it=arr[pos].begin();it!=arr[pos].end();it++){
            if((*it) == key) return;
        }
        arr[pos].push_back(key);
    }
    
    void remove(int key) {
        int pos=key%size;
        for(auto it=arr[pos].begin();it!=arr[pos].end();it++){
            if((*it) == key){
                arr[pos].erase(it);
                return;
            }
        }
    }
    
    /** Returns true if this set contains the specified element */
    bool contains(int key) {
        int pos=key%size;
        for(auto it=arr[pos].begin();it!=arr[pos].end();it++){
            if((*it) == key) return true;
        }
        return false;
    }
};
```



# 33

搜索旋转排序数组

有个坑：当旋转点为0时其实没有旋转

二分搜索：左右两半总有一半是升序的，可以判断在不在这一段里

```c++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int n=nums.size();
        if(!n) return -1;
        if(n==1) return nums[0]==target? 0:-1;

        int l=0,r=n-1;
        int mid;

        while(l<=r){
            mid=(l+r)/2;
            if(nums[mid]==target) return mid;

            if(nums[l]<=nums[mid]){
                if(nums[l]<=target && target<nums[mid]){
                    r=mid-1;
                }else{
                    l=mid+1;
                }
            }else{
                if(nums[mid]<target && target<=nums[r]){
                    l=mid+1;
                }else{
                    r=mid-1;
                }
            }
        }
        return -1;
    }
};
```



# 706

设计哈希映射

将单一的key改为键值对

type: pair<int,int>

创建：make_pair(key,value)

访问元素：.first,.second

```c++
class MyHashMap {
public:
    vector<list<pair<int,int>>> arr;
    static const int size=1001;
    /** Initialize your data structure here. */
    MyHashMap() {
        arr.resize(size);
    }
    
    /** value will always be non-negative. */
    void put(int key, int value) {
        int pos=key%size;
        for(auto it=arr[pos].begin();it!=arr[pos].end();it++){
            if((*it).first == key){
                (*it).second=value;
                return;
            }
        }
        arr[pos].push_back(make_pair(key,value));
    }
    
    /** Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key */
    int get(int key) {
        int pos=key%size;
        for(auto it=arr[pos].begin();it!=arr[pos].end();it++){
            if((*it).first == key) return (*it).second;
        }
        return -1;
    }
    
    /** Removes the mapping of the specified value key if this map contains a mapping for the key */
    void remove(int key) {
        int pos=key%size;
        for(auto it=arr[pos].begin();it!=arr[pos].end();it++){
            if((*it).first == key){
                arr[pos].erase(it);
                return;
            }
        }
    }
};
```



# 54

螺旋矩阵

一层一层

遍历

```c++
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        int m=matrix.size();
        int n=matrix[0].size();
        int num=m*n;
        vector<int> ans;

        int top=0,bottom=m-1;
        int left=0,right=n-1;
        
        while(num){
            for(int j=left;j<=right && num;j++){
                ans.push_back(matrix[top][j]);
                num--;
            }
            top++;
            for(int j=top;j<=bottom && num;j++){
                ans.push_back(matrix[j][right]);
                num--;
            }
            right--;
            for(int j=right;j>=left && num;j--){
                ans.push_back(matrix[bottom][j]);
                num--;
            }
            bottom--;
            for(int j=bottom;j>=top && num;j--){
                ans.push_back(matrix[j][left]);
                num--;
            }
            left++;
        }
        return ans;
    }
};
```



# 59

螺旋矩阵II

跟上一道差不多，不过矩阵是正方形的

```
class Solution {
public:
    vector<vector<int>> generateMatrix(int n) {
        vector<vector<int>> m(n,vector<int>(n));

        int len=n*n;
        int b=0,el=1;

        while(el<=len){
            for(int i=b;i<n-b && el<=len;i++){
                m[b][i]=el;
                el++;
            }
            for(int i=b+1;i<n-b && el<=len;i++){
                m[i][n-b-1]=el;
                el++;
            }
            for(int i=n-b-2;i>=b && el<=len;i--){
                m[n-b-1][i]=el;
                el++;
            }
            for(int i=n-b-2;i>=b+1 && el<=len;i--){
                m[i][b]=el;
                el++;
            }
            b++;
        }
        return m;
    }
};
```



# 115

不同的子序列

动态规划

假设字符串 $s$ 和 $t$ 的长度分别为 $m$ 和 $n$。如果 $t$ 是 $s$ 的子序列，则 $s$ 的长度一定大于或等于 $t$ 的长度，如果 $m<n$，则 $t$ 一定不是 $s$ 的子序列，直接返回 $0$。

创建二维数组 $dp$，其中 $dp[i][j]$ 表示在 $s[i:]$ 的子序列中 $t[j:]$ 出现的个数。

上述表示中，$s[i:]$ 表示 $s$ 从下标 $i$ 到末尾的子字符串，$t[j:] $表示 $t$ 从下标 $j$ 到末尾的子字符串。

考虑动态规划的边界情况：

当 $j=n$ 时，$t[j:] $为空字符串，由于空字符串是任何字符串的子序列，因此对任意 $0 \le i \le m$，有 $dp[i][n]=1$

当 $i=m$ 且 $j<n $时，$s[i:] $为空字符串，$t[j:]$ 为非空字符串，由于非空字符串不是空字符串的子序列，因此对任意 $0 \le j<n$，有 $dp[m][j]=0$。

当 $i<m$ 且$ j<n$ 时，考虑 $dp[i][j]$ 的计算：

- 当 $s[i]=t[j]$ 时，$dp[i][j] $由两部分组成：

  如果 $ s[i]$ 和 $t[j]$ 匹配，则考虑 $t[j+1:]$ 作为 $s[i+1:]$ 的子序列，子序列数为 $dp[i+1][j+1]$；

  如果 $ s[i]$ 不和 $t[j]$ 匹配，则考虑 $t[j:] $作为 $s[i+1:] $的子序列，子序列数为 $dp[i+1][j]$。

  因此当 $s[i]=t[j]$ 时，有 $dp[i][j]=dp[i+1][j+1]+dp[i+1][j]$。

- 当 $s[i] \ne t[j]$时，$s[i]$ 不能和 $t[j]$ 匹配，因此只考虑 $t[j:]$ 作为 $s[i+1:]$的子序列，子序列数为 $dp[i+1][j]$。


```c++
class Solution {
public:
    int numDistinct(string s, string t) {
        int m=s.length();
        int n=t.length();

        if(m<n) return 0;

        vector<vector<long>> dp(m+1,vector<long>(n+1));
        for(int i=0;i<=m;i++){
            dp[i][n]=1;
        }

        for(int i=m-1;i>=0;i--){
            for(int j=n-1;j>=0;j--){
                if(s[i]==t[j]){
                    dp[i][j]=dp[i+1][j+1]+dp[i+1][j];
                }else{
                    dp[i][j]=dp[i+1][j];
                }
            }
        }

        return dp[0][0];
    }
};
```



# 92

反转链表

方法一：将需要反转的部分反转之后，再和剩余部分拼接起来，缺点：需要两次遍历

方法二：只遍历一次，在需要反转的区间内，将每一个节点插入反转的起始点

![image.png](https://pic.leetcode-cn.com/1615105296-bmiPxl-image.png)

```c++
class Solution {
public:
    ListNode* reverseBetween(ListNode* head, int left, int right) {
        ListNode* dummy=new ListNode(0, head);
        ListNode* pre=dummy;

        for(int i=0;i<left-1;i++){
            pre=pre->next;
        }

        ListNode *curr, *nex;
        curr=pre->next;
        for(int i=0;i<right-left;i++){
            nex=curr->next;
            curr->next=nex->next;
            nex->next=pre->next;
            pre->next=nex;
        }

        return dummy->next;
    }
};
```



# 173

二叉树搜索迭代器

中序遍历一遍存到数组里

```c++
class BSTIterator {
private:
    vector<int> vec;
    int it=0;
    void traverse(TreeNode* root){
        if(!root) return;
        traverse(root->left);
        vec.push_back(root->val);
        traverse(root->right);
    }
public:
    BSTIterator(TreeNode* root) {
        traverse(root);
    }
    
    int next() {
        return vec[it++];
    }
    
    bool hasNext() {
        return it<vec.size();
    }
};
```



# 300

最长递增子序列

dp[i]：以nums[i]结尾的最长递增子序列

dp数组里最大的那个就是所求

```c++
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int n=nums.size();
        vector<int> dp(n,1);

        for(int i=1;i<n;i++){
            for(int j=i-1;j>=0;j--){
                if(nums[j]<nums[i]){
                    dp[i]=max(dp[i],dp[j]+1);
                }
            }
        }

        return *max_element(dp.begin(),dp.end());
    }
};
```

贪心+二分查找

要使上升子序列尽可能的长，则要让序列上升得尽可能慢，每次在上升子序列最后加上得数要尽可能小

用dp[i]维护长度为i的最长上升子序列的最小值，用len记录目前最长上升子序列的长度，dp[i]是单调递增的

```c++
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int n=nums.size();
        vector<int> dp(n+1,0);
        dp[1]=nums[0];

        int len=1;
        for(int i=1;i<n;i++){
            if(nums[i]>dp[len]){
                dp[++len]=nums[i];//加在最后面
            }else{
                auto pos=lower_bound(dp.begin()+1,dp.begin()+len+1,nums[i]);
                //二分查找比nums[i]大的数，并把它换掉
                *(pos)=nums[i];
            }
        }
        return len;
    }
};
```



# 190

颠倒二进制位

逐位操作

```c++
class Solution {
public:
    uint32_t reverseBits(uint32_t n) {
        uint32_t ans=0;
        for(int i=0;i<32;i++){
            ans<<=1;
            ans+=(n%2);
            n>>=1;
        }
        return ans;
    }
};
```

位运算分治：妙啊

递归执行翻转，左半部分放到右边，右半部分放到左边

可以利用位掩码和移位运算

对于最底层，要交换所有奇偶位

```c++
class Solution {
private:
    const uint32_t M1 = 0x55555555; // 01010101010101010101010101010101
    const uint32_t M2 = 0x33333333; // 00110011001100110011001100110011
    const uint32_t M4 = 0x0f0f0f0f; // 00001111000011110000111100001111
    const uint32_t M8 = 0x00ff00ff; // 00000000111111110000000011111111

public:
    uint32_t reverseBits(uint32_t n) {
        n = n >> 1 & M1 | (n & M1) << 1;
        n = n >> 2 & M2 | (n & M2) << 2;
        n = n >> 4 & M4 | (n & M4) << 4;
        n = n >> 8 & M8 | (n & M8) << 8;
        return n >> 16 | n << 16;
    }
};
```

