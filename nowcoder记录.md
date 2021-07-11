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



# KY155

To Fill or Not to Fill

贪心算法：

按加油站的距离升序考虑，需要决定的是在这个加油站加多少油？下一个到达的加油站应该是哪个？

若该加油站开始，往后最大行驶距离（即满油行驶距离或者到达终点）中：

①有加油站的价格比当前便宜，则当前加油站只需要从当前油量把油加到，能够行驶到第一个比当前加油站便宜的加油站所需要的油量即可

②都比当前加油站油价贵，那么只需要从当前油量把油加满即可，然后行驶到的下一个加油站应该是最大行驶距离内，油价最便宜的那一个加油站

③若没有加油站了，则判断当前最大行驶距离是否能达到目的地

```c++
#include <iostream>
#include <algorithm>

using namespace std;

int cmax, d, davg, n;

struct station {
    double pi;
    int di;

    bool operator<(const station &b) const {
        return pi < b.pi;
    }
} a[501];

int main() {
    double sum;
    while (cin >> cmax >> d >> davg >> n) {
        for (int i = 0; i < n; i++) {
            cin >> a[i].pi >> a[i].di;
        }
        sort(a, a + n); //按照油价升序排列
        sum = 0;
        bool flag[30001] = {0}; //表示该距离能不能到达
        int maxd = cmax * davg; //满油能行驶的最远距离
        int tmp, cnt;
        for (int i = 0; i < n; i++) {
            tmp = (a[i].di + maxd) < d ? maxd : d - a[i].di; //当前的最大行驶距离
            cnt = 0; //当前需要加的油
            for (int j = a[i].di; j < a[i].di + tmp; j++) {
                if (!flag[j]) { //true说明已经被之前的加油站（便宜的）覆盖了
                    flag[j] = true;
                    cnt++;
                }
            }
            sum += cnt / double(davg) * a[i].pi; //花的钱
        }

        int i;
        for (i = 0; i < d; i++) {
            if (!flag[i]) {
                break; //有的路段行驶不到
            }
        }

        if (i == d) {
            printf("%.2f\n", sum);
        } else {
            printf("The maximum travel distance = %.2f\n", double(i));
        }
    }

    return 0;
}
```



# KY73

合唱队形

dp求从前往后的最长递增子序列和从后往前的最长递增子序列

注意子序列长度最小为1！自己本身

```c++
#include <iostream>

using namespace std;

int n;
int t[101],inc[101],de[101];

int main(){
    while(cin>>n){
        for(int i=0;i<n;i++){
            cin>>t[i];
        }
        //从前往后
        for(int i=0;i<n;i++){
            inc[i]=1;
            for(int j=0;j<i;j++){
                if(t[i]>t[j]){
                    inc[i]=max(inc[i],inc[j]+1);
                }
            }
        }
        //从后往前
        for(int i=n-1;i>=0;i--){
            de[i]=1;
            for(int j=n-1;j>i;j--){
                if(t[i]>t[j]){
                    de[i]=max(de[i],de[j]+1);
                }
            }
        }
        int k=0;
        for(int i=0;i<n;i++){
            k=max(k,inc[i]+de[i]-1);//i这个位置重复算了，要减1
        }
        cout<<n-k<<endl;
    }
    
    return 0;
}
```


# KY14

最小邮票数

01背包dp，dp数组维护该面值所需最小邮票数

空间优化，从后往前更新

```c++
#include <iostream>

using namespace std;

int m,n;
int v[20];
int dp[100]={0};

int main(){
    while(cin>>m>>n){
        for(int i=0;i<n;i++){
            cin>>v[i];
        }
        for(int i=0;i<n;i++){
            for(int j=m;j>=v[i];j--){
                if(dp[j-v[i]]){ //j-v[i]可以凑出来
                    if(dp[j]) //j之前可以凑出来，取较小值
                        dp[j]=min(dp[j],dp[j-v[i]]+1);
                    else //j之前凑不出来，直接更新
                        dp[j]=dp[j-v[i]]+1;
                }
            }
            dp[v[i]]=1;
        }
        cout<<dp[m]<<endl;
    }
    
    return 0;
}
```



# KY12

玛雅人的密码

用BFS去尝试每种可能的交换

用map把出现过的字符串都记下来

```c++
#include <iostream>
#include <string>
#include <map>
#include <queue>

using namespace std;

int bfs(string s){
    map<string, int> M;
    queue<string> Q;
    
    Q.push(s);
    M[s]=0;//初始的交换次数是0
    while(!Q.empty()){
        string str=Q.front();
        Q.pop();
        for(int i=0;i<str.size()-1;i++){
            string newstr=str;
            swap(newstr[i],newstr[i+1]);
            if(M.find(newstr)==M.end()){//没有在map中，没尝试过
                M[newstr]=M[str]+1;//新的字符串比旧的交换次数+1
                if(newstr.find("2012")!=string::npos) return M[newstr];//找到密码
                Q.push(newstr);
            }else{
                continue;//在map中找到表示已经尝试过
            }
        }
    }
    
    return -1;
}

int main(){
    int n;
    string s;
    
    while(cin>>n){
        cin>>s;
        if(s.find("2012")!=string::npos) cout<<"0"<<endl;//字符串里本身就有
        else{
            cout<<bfs(s)<<endl;
        }
    }
    
    return 0;
}
```

