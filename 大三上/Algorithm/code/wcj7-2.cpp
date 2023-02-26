#include <iostream>

static int inf=10000;
const int N = 1e5;
int size;//���ڼ�¼��̳����еĳ��ȣ�����������
int a[N], b[N], c[N];

using namespace std;

int** DoubleLength(int a[],int b[],int n1,int n2)
{
    int maxn=max(n1,n2)+1;
    int **dp=new int*[maxn];//�����й��������еĳ�������ɵ�����
    for(int i=0;i<maxn;i++)
    {
        dp[i]=new int[maxn];
        for(int j=0;j<maxn;j++)
        dp[i][j]=inf;//��ʼ������
    }
    for(int i=0;i<maxn;i++) dp[i][0]=dp[0][i]=i; //��ʼ��dp
    for(int i=1;i<=n1;i++) //״̬ת��
    {
        for(int j=1;j<=n2;j++)
        {
            if (a[i-1]==b[j-1]) dp[i][j]=dp[i-1][j-1]+1;//�ַ����ʱ��Ҫ�࿼��һ�����
            dp[i][j] = min(min(dp[i-1][j]+1,dp[i][j-1]+1),dp[i][j]);
        }
    }
    return dp;
}

int* DoubleCreate(int a[],int b[],int n1,int n2) //���������еĹ���������
{
    int **dp=DoubleLength(a,b,n1,n2);
    int *ans=new int[dp[n1][n2]];//����������
    int count=dp[n1][n2];//���ڱ�������
    int size=dp[n1][n2];//��¼���鳤��
    while(n1>0&&n2>0) //ͨ�����������г��ȵ��˳�����
    {
        if(dp[n1][n2]==dp[n1-1][n2-1]+1&&a[n1-1]==b[n2-1])//���ʱ�����
        {
            ans[--count]=a[n1-1];
            n1--;
            n2--;
        }
        else if(dp[n1][n2]==dp[n1-1][n2]+1) //�����������a
        {
            ans[--count]=a[n1-1];
            n1--;
        }
        else if(dp[n1][n2]==dp[n1][n2-1]+1) //�����������b
        {
            ans[--count]=b[n2-1];
            n2--;
        }
    }
    while(n1>0) //a����ʣ�����
    {
        ans[--count]=a[n1-1];
        n1--;
    }
    while(n2>0) //b����ʣ�����
    {
        ans[--count]=b[n2-1];
        n2--;
    }
    return ans;
}

int* TripleCreate(int a[],int b[],int c[],int n1,int n2,int n3)
{
    int**dp12=DoubleLength(a,b,n1,n2);
    int**dp13=DoubleLength(a,c,n1,n3);
    int**dp23=DoubleLength(b,c,n2,n3);//�ֱ��������֮��Ĺ��������г�������
    int maxsize=max(max(n1,n2),n3)+1;
    int dp[maxsize][maxsize][maxsize];//�����й������������鳤������
    for(int i=0; i<=n1; i++) //��ʼ������
        for(int j=0; j<=n2; j++)
            for(int k=0; k<=n3; k++)
            {
                dp[i][j][k]=inf;
                if(i==0) dp[i][j][k]=dp23[j][k];
                else if(j==0) dp[i][j][k]=dp13[i][k];
                else if(k==0) dp[i][j][k]=dp12[i][j];//��Ե����˻�Ϊ�������е����
            }
    for(int i=1; i<=n1; i++) //��ά�������������鳤�ȵ�״̬ת�ƹ�ϵ��һ����7�����
        for(int j=1; j<=n2; j++)
            for(int k=1; k<=n3; k++)
            {
                if(a[i-1]==b[j-1]&&b[j-1]==c[k-1]) //�����������ʱ
                    dp[i][j][k]=dp[i-1][j-1][k-1]+1;
                if(a[i-1]==b[j-1])
                    dp[i][j][k]=min(dp[i-1][j-1][k]+1,dp[i][j][k]);
                if(a[i-1]==c[k-1])
                    dp[i][j][k]=min(dp[i-1][j][k-1]+1,dp[i][j][k]);
                if(b[j-1]==c[k-1])
                    dp[i][j][k]=min(dp[i][j-1][k-1]+1,dp[i][j][k]);//���������ʱ
                dp[i][j][k]=min(min(min(dp[i-1][j][k]+1,dp[i][j][k]),dp[i][j-1][k]+1),dp[i][j][k-1]+1);
            }
    int*ans=new int[dp[n1][n2][n3]];//����������
    int count=dp[n1][n2][n3];//���ڱ�������
    size=dp[n1][n2][n3];
    int i=n1,j=n2,k=n3;
    while(i>0&&j>0&&k>0)//ͨ�����������鳤�ȷ���õ�������
    {
        if(dp[i][j][k]==dp[i-1][j-1][k-1]+1&&a[i-1]==b[j-1]&&b[j-1]==c[k-1])//���������ʱ
        {
            ans[--count]=a[i-1];
            i--;
            j--;
            k--;
        }
        else if(dp[i][j][k]==dp[i-1][j-1][k]+1&&a[i-1]==b[j-1])//���������ʱ
        {
            ans[--count]=a[i-1];
            i--;
            j--;
        }
        else if(dp[i][j][k]==dp[i-1][j][k-1]+1&&a[i-1]==c[k-1])
        {
            ans[--count]=a[i-1];
            i--;
            k--;
        }
        else if(dp[i][j][k]==dp[i][j-1][k-1]+1&&b[j-1]==c[k-1])
        {
            ans[--count]=b[j-1];
            j--;
            k--;
        }
        else//������������ͬʱ
        {
            int minimal=min(min(dp[i-1][j][k],dp[i][j-1][k]),dp[i][j][k-1]);
            if(dp[i-1][j][k]==minimal)
            {
                ans[--count]=a[i-1];
                i--;
            }
            else if(dp[i][j-1][k]==minimal)
            {
                ans[--count]=b[j-1];
                j--;
            }
            else
            {
                ans[--count]=c[k-1];
                k--;
            }
        }
    }
    if(i==0&&count>0)//һ������Ϊ��ʱ���˻����������еĹ��������е����
    {
        int*ans1=DoubleCreate(b,c,j,k);
        for(int t=0; t<count; t++) ans[t]=ans1[t];
    }
    if(j==0&&count>0)
    {
        int*ans1=DoubleCreate(a,c,i,k);
        for(int t=0; t<count; t++) ans[t]=ans1[t];
    }
    if(k==0&&count>0)
    {
        int*ans1=DoubleCreate(a,b,i,j);
        for(int t=0; t<count; t++) ans[t]=ans1[t];
    }
    return ans;
}
int main()
{
    freopen("7-2-num.in", "r", stdin);
    int n, n1, n2, n3;
    cin >> n;
    n1 = n2 = n3 = n;
    for (int i = 0; i < n; i++) cin >> a[i];
    for (int i = 0; i < n; i++) cin >> b[i];
    for (int i = 0; i < n; i++) cin >> c[i];
    int* ans=TripleCreate(a,b,c,n1,n2,n3);
    cout<<"My Answer:"<<endl;
    string s;
    for(int i=0;i<size;i++) cout<<ans[i]<<' ', s.push_back(ans[i] + 'a');
    cout<<endl;
    cout << "string: " << s << '\n';
    cout << "len: " << size << '\n';
    return 0;
}
