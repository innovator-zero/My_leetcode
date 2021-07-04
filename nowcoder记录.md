# KY16

求root(N,k)

数学推导要思考

![image-20210704181114526](C:\Users\Yuxiang Lu\AppData\Roaming\Typora\typora-user-images\image-20210704181114526.png)

然后就是快速幂

```c++
#include <iostream>

using namespace std;

long long x, y;
int k;

int main() {
    cin >> x >> y >> k;

    int res = 1;

    while (y) {
        if (y & 1) {
            res = (res * x) % (k - 1);
        }
        x = (x % (k - 1)) * (x % (k - 1));
        x %= (k - 1);
        y >>= 1;
    }

    if (!res) res = k - 1;
    cout << res << endl;

    return 0;
}
```

# KY105

整除问题

A%B==0说明B的质因数是A的质因数的子集

主要考察了：1.求素数，2.质因数分解

因为数最大是1000，所以素因子不会超过sqrt(1000)，这里近似取的40

```c++
#include <iostream>
#include <map>
#include <cmath>
#include <cstring>

using namespace std;

const int MAX = 40;
bool isPrime[MAX];
int prime[MAX];
int cnt = 0;

void factor(map<int, int> &m, int x) {
    //对x做质因数分解
    int s = int(sqrt(x)); //最大的素因子不会超过sqrt(x)
    for (int i = 0; i < cnt && prime[i] <= s; i++) {
        //考虑每个素数
        while (x % prime[i] == 0) {
            m[prime[i]]++;
            x /= prime[i];
        }
    }
    if (x != 1) {
        m[x]++; //如果x剩下的不是1，那说明x必定是个素数，这里的x就等于原始的x，它唯一的素因子就是自己
    }
}

int main() {
    int n, a;
    cin >> n >> a;

    //素数筛求MAX以内的所有数是不是素数
    memset(isPrime, true, sizeof(isPrime));
    for (int i = 2; i < MAX; i++) {
        if (isPrime[i]) {
            for (int j = i * i; j < MAX; j += i) {
                isPrime[j] = false;
            }
        }
    }

    //把素数放到prime数组里
    for (int i = 2; i < MAX; i++) {
        if (isPrime[i])
            prime[cnt++] = i;
    }


    map<int, int> map_n, map_a;

    for (int i = 2; i <= n; i++) {
        factor(map_n, i);
    }
    factor(map_a, a);

    int min_k = 1e7;
    for (int i = 0; i <= 1000; i++) {
        if (map_a[i])
            min_k = min(min_k, int(map_n[i] / map_a[i]));
    }

    cout << min_k << endl;

    return 0;
}
```



# KY212

二叉树的遍历

用前序遍历和中序遍历生成后序遍历

把左右子树的部分分割出来

前序: root, left, right

中序: left, root, right

对左右子树递归调用，再输出root

```c++
#include <iostream>
#include <string>

using namespace std;

void post(string& pre, string& mid){
    if(pre.length()==0) return;
    char root=pre[0];
    int rootindex=mid.find(root);//也是左子树的字符长度
    string leftpre=pre.substr(1,rootindex);
    string leftmid=mid.substr(0,rootindex);
    string rightpre=pre.substr(rootindex+1);
    string rightmid=mid.substr(rootindex+1);
    post(leftpre, leftmid);
    post(rightpre, rightmid);
    cout<<root;
}

int main(){
    string pre,mid;
    
    while(cin>>pre>>mid){
        post(pre, mid);
        cout<<endl;
    }
    
    return 0;
}
```



# KY188

哈夫曼树

一个结点的权值=路径长度（根结点到该结点上的结点数量）=祖先结点的数量

也等于建树过程中被选中的次数，每选中一次多一个祖先结点

```c++
#include <iostream>
#include <queue>

using namespace std;

int main(){
    int n,weight,tmp;
    priority_queue<int, vector<int>,greater<int>> minheap;//最小堆的写法
    while(cin>>n){
        weight=0;
        for(int i=0;i<n;i++){
            cin>>tmp;
            minheap.push(tmp);
        }
    
        int a,b;
        while(minheap.size()!=1){
            a=minheap.top();
            minheap.pop();
            b=minheap.top();
            minheap.pop();
            weight+=(a+b);//每次选中权值+1
            minheap.push(a+b);
        }
        cout<<weight<<endl;
        minheap.pop();
    }

    return 0;
}
```

