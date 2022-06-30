// ConsoleApplication1.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <vector>
#include <string.h>
#include <stdint.h>
#include <algorithm>
#include <stack>
#include <queue>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <string>

#include "jump.h"
using namespace std;
class Tnode {
	//树节点类
public:
	Tnode *right;
	Tnode *left;
	int val;
	Tnode(int v) {
		val = v;
		right = nullptr;
		left = nullptr;
	}
};
class RNode {
public:
	int val;
	RNode* next;
	RNode* random;

	RNode(int _val) {
		val = _val;
		next = NULL;
		random = NULL;
	}
};
class info {
	//树节点类
public:
	int Dis;
	int Hight;
	info(int v, int r) {
		Dis = v;
		Hight = r;
	}
};

info press(Tnode* head) {
	//计算一个树两个结点的最大距离
	//返回最大距离和树的最大深度
	if (head == nullptr)return info(0, 0);
	info p1 = press(head->left);
	info p2 = press(head->right);
	//计算info
	int d1 = p1.Dis;
	int d2 = p2.Dis;
	int d3 = p1.Hight + p2.Hight + 1;
	int maxD = max(d1, max(d2, d3));
	int maxH = max(p1.Hight, p2.Hight) + 1;
	return info(maxD, maxH);
}

string strTrans(string str) {
	//向字符串中添加 #
	string s;
	for (int i = 0; i < str.size(); i++) {
		if (i == 0)s.push_back('#');
		s.push_back(str[i]);
		s.push_back('#');
	}
	return s;
}

vector<int> getNext(string s) {
	vector<int> next(s.size());
	next[0] = -1;
	int j = 0, k = -1;
	while (j < s.size() - 1) {
		if (k == -1 || s[j] == s[k]) {
			j++;
			k++;
			next[j] = k;
		}
		//不理解？？？
		else k = next[k];
	}
	return next;
}

int KMP(string s, string p) {
	//传入主串和模式串
	//i j 分别为两个串的下标
	int i = 0, j = 0;
	vector<int> next = getNext(p);
	while (i < s.size() && j < p.size()) {
		if (i == -1 || s[i] == p[j]) {
			i++;
			j++;
		}
		else j = next[j];
	}
	if (j == p.size())return i - j;
	else return -1;
}

int manacher(string str) {
	//计算一个字符串中的最大回文子串
	string s = strTrans(str);//123-->#1#2#3#
	vector<int> pr(s.size());//回文半径数组
	//回文半径数组 pr[i]:下标 i 的元素在字符串中以 i 为中心的回文串的长度一半+1
	int r = -1;//回文右边界 再往右的位置
	int c = -1;//回文中心
	int maxR = INT_MIN;//扩出来的最大值
	for (int i = 0; i < s.size(); i++) {
		pr[i] = r > i ? min(r - i, pr[2 * c - i]) : 1;
		while (i + pr[i]<s.size() && i - pr[i]>-1) {
			if (s[i + pr[i]] == s[i - pr[i]]) {
				pr[i]++;
			}
			else {
				break;
			}
		}
		if (i + pr[i] > r) {
			r = i + pr[i];
			c = i;
		}
		maxR = max(maxR, pr[i]);
	}
	return maxR - 1;//因为字符串中添加了# # #
}

class Dlist {
	//双向链表
public:
	int val;
	Dlist *pre;
	Dlist *next;
	Dlist(int v) {
		val = v;
		pre = nullptr;
		next = nullptr;
	}
};

void RemoveDlist(Dlist *node) {
	//从一个双向链表中删除这个结点
	node->pre->next = node->next;
	node->next->pre = node->pre;
}

//括号嵌套//
string compress(string str) {
	// write code here
	//出现括号 优先 栈
	stack<string> s;
	stack<int> nums;
	int num = 0;
	string res = "";
	for (int i = 0; i < str.size(); i++) {
		if (str[i] >= '0'&&str[i] <= '9') {
			num = num * 10 + (str[i] - '0');
		}
		else if (str[i] >= 'A'&&str[i] <= 'Z') {
			res.push_back(str[i]);
		}
		else if (str[i] == '|') {
			nums.push(num);
			s.push(res);
			num = 0;
			res = "";
		}
		else if (str[i] == ']') {
			int t = nums.top();
			nums.pop();
			for (int j = 0; j < t; j++) {
				s.top() = s.top() + res;
			}
			//之后若还是字母，就会直接加到res之后，因为它们是同一级的运算
			//若是左括号，res会被压入strs栈，作为上一层的运算
			res = s.top();
			s.pop();
		}
	}
	return res;
}

map<string, int> compress_h(string& str, int index) {
	int tmp_num = 0;
	string tmp_s;
	while (index < str.size() && str[index] != ']') {
		if (str[index] >= '0'&&str[index] <= '9') {
			tmp_num = tmp_num * 10 + str[index] - '0';
			index++;
		}
		else if (str[index] != '[') {
			tmp_s.push_back(str[index]);
			index++;
		}
		else {
			map<string, int> t = compress_h(str, index + 1);
			for (int i = 0; i < tmp_num; i++) {
				tmp_s = tmp_s + t.begin()->first;
			}
			tmp_num = 0;
			index = t.begin()->second + 1;
		}
	}
	return { {tmp_s,index} };
}
/*//单调栈！！！！
public int[] findBuilding (int[] heights) {
	int n = heights.length;
	int[] ans = new int[n];
	LinkedList<Integer> stack1 = new LinkedList<>(), stack2 = new LinkedList<>();
	Arrays.fill(ans, 1);
	// 往左看，也就是要得到每个数左边有多少递增的（不要求连续递增，不包括自身）
	for(int i = 0;i < n-1;i++) {
		while(!stack1.isEmpty() && heights[i] >= stack1.getFirst()) {
			stack1.removeFirst();
		}
		stack1.addFirst(heights[i]);
		ans[i+1] += stack1.size();
	}
	// 往右看，也就是要得到每个数右边有多少递增的
	for(int i = n-1;i > 0;i--) {
		while(!stack2.isEmpty() && heights[i] >= stack2.getFirst()) {
			stack2.removeFirst();
		}
		stack2.addFirst(heights[i]);
		ans[i-1] += stack2.size();
	}
	return ans;
}
*/

vector<vector<int>> monotoneStack(vector<int>& nums) {
	//查找数组中每个元素左边和右边第一个比自己大的数，如果没有就填充为INT_MIN
	//针对输入数据不存在重复的情况
	//栈 从底到顶是单调减小的
	stack<int> s;
	vector<vector<int>> ans(nums.size(), vector<int>(2));
	for (int i = 0; i < nums.size(); i++) {
		if (s.empty() || nums[i] < nums[s.top()]) {
			s.push(i);
		}
		else {
			while (!s.empty() && nums[i] > nums[s.top()]) {
				int index = s.top();
				s.pop();
				int leftMax = s.empty() == true ? INT_MIN : nums[s.top()];
				int rightMax = nums[i];
				ans[index] = { leftMax,rightMax };
			}
			s.push(i);
		}
	}
	//清算,此时栈中的所有元素右边都没有比它更大的了
	while (!s.empty()) {
		int index = s.top();
		s.pop();
		int leftMax = s.empty() == true ? INT_MIN : nums[s.top()];
		int rightMax = INT_MIN;
		ans[index] = { leftMax,rightMax };
	}
	return ans;
}

vector<int> bfs(vector<vector<char>>& g, int x, int y, int dx, int dy, vector<vector<bool>>& used) {
	queue<vector<int>> q;
	q.push({ x,y });

	int step = 0;
	while (!q.empty()) {
		int size = q.size();
		for (int i = 0; i < size; i++) {
			vector<int> pp = q.front();
			q.pop();
			int n = pp[0];
			int m = pp[1];
			if (n + 1 < used.size() && used[n + 1][m] == false && (g[n + 1][m] == '.' || g[n + 1][m] == 'S')) {
				q.push({ n + 1,m });
				int tx = n, ty = m - 1;
				if ((tx - 1 == dx && ty == dy) || (tx + 1 == dx && ty - 1 == dy) || (tx == dx && ty - 1 == dy) || (tx == dx && ty + 1 == dy))
					return { tx,ty,step,1 };
			}
			if (n - 1 >= 0 && used[n - 1][m] == false && (g[n - 1][m] == '.' || g[n - 1][m] == 'S')) {
				q.push({ n - 1,m });
				int tx = n, ty = m - 1;
				if ((tx - 1 == dx && ty == dy) || (tx + 1 == dx && ty - 1 == dy) || (tx == dx && ty - 1 == dy) || (tx == dx && ty + 1 == dy))
					return { tx,ty,step,1 };
			}
			if (m + 1 < used[0].size() && used[n][m + 1] == false && (g[n][m + 1] == '.' || g[n][m + 1] == 'S')) {
				q.push({ n,m + 1 });
				int tx = n, ty = m - 1;
				if ((tx - 1 == dx && ty == dy) || (tx + 1 == dx && ty - 1 == dy) || (tx == dx && ty - 1 == dy) || (tx == dx && ty + 1 == dy))
					return { tx,ty,step,1 };
			}
			if (m - 1 >= 0 && used[n][m - 1] == false && (g[n][m - 1] == '.' || g[n][m - 1] == 'S')) {
				q.push({ n,m - 1 });
				int tx = n, ty = m - 1;
				if ((tx - 1 == dx && ty == dy) || (tx + 1 == dx && ty - 1 == dy) || (tx == dx && ty - 1 == dy) || (tx == dx && ty + 1 == dy))
					return { tx,ty,step,1 };
			}
			used[n][m] = true;
			step++;
		}
	}
	return { 0,0,step,0 };
	/*
	输入输出测试
	vector<vector<char>> g = {{'#','.','.','.','.'},{'#','#','#','#','.'},{'F','S','.','.','.'}};
	vector<vector<int>> node = {{0,0},{1,2},{1,1},{2,0}};
	int step = 0;
	int sx = 2,sy = 1;
	for(int i = 0;i<node.size();i++){
		vector<vector<bool>> used(3,vector<bool>(5,false));
		vector<int> tmp = bfs(g,sx,sy,node[i][0],node[i][1],used);
		sx = tmp[0],sy = tmp[1];
		if(tmp[3]==0)step = -1;
		else step = step + tmp[2];
	}
	cout<<step;
	*/
}

class G_node {
	//图的结点 包括结点数值，结点可以到达的下个结点，结点所属的边
public:
	int val;
	int from;
	int to;
	vector<G_node*> nexts;
	G_node() {}//需要一个默认构造函数
	G_node(int v) {
		val = v;
		from = 0;
		to = 0;
	}

};

class G_edge {
	//图的边 包括边的权值，边的起点和终点
public:
	int weight;
	G_node* from;
	G_node* to;
	G_edge(int w, G_node* from, G_node* to) {
		weight = w;
		this->from = from;
		this->to = to;
	}
};

class G {
	//图 包括所有的结点，所有的边
public:
	map<int, G_node*> nodes;
	unordered_set<G_edge*> edges;
};

/*//建图 1--0 1--2 2--3 3--0 0--2 1--4 4--5 无向图
	vector<vector<int>> path = {{1,0},{1,2},{2,3},{3,0},{0,2},{1,4},{4,5},{2,6}};
	G myG;
   for(int i = 0;i<path.size();i++){
	   int from = path[i][0];
	   int to = path[i][1];
	   if(!myG.nodes.count(from))myG.nodes.insert({from,new G_node(from)});
	   if(!myG.nodes.count(to))myG.nodes.insert({to,new G_node(to)});
	   G_node* fromNode = myG.nodes[from];
	   G_node* toNode = myG.nodes[to];
	   //from-->to
	   G_edge* newEdge = new G_edge(1,fromNode,toNode);
	   fromNode->nexts.push_back(toNode);
	   myG.edges.insert(newEdge);
	   //to-->from
	   newEdge = new G_edge(1,toNode,fromNode);
	   toNode->nexts.push_back(fromNode);
	   myG.edges.insert(newEdge);
   }*/

void Gdfs(G_node* from) {
	//从一个点出发，深度优先遍历，找到所能到达的所有点
	stack<G_node*> help;
	unordered_set<G_node*> s;
	help.push(from);
	s.insert(from);
	cout << from->val << "->";
	while (!help.empty()) {
		G_node* cur = help.top();
		help.pop();
		for (auto n : cur->nexts) {
			if (!s.count(n)) {
				//注意两个 push 的顺序
				help.push(cur);
				help.push(n);
				s.insert(n);
				cout << n->val << "->";
				break;
			}
		}
	}
	cout << endl;
}

void Gbfs(G_node* from) {
	//从一个点出发，广度优先遍历，找到所能到达的所有点
	queue<G_node*> help;
	unordered_set<G_node*> s;
	help.push(from);
	s.insert(from);
	while (!help.empty()) {
		G_node* cur = help.front();
		help.pop();
		cout << cur->val << "->";
		for (auto n : cur->nexts) {
			if (!s.count(n)) {
				s.insert(n);
				help.push(n);
			}
		}

	}
}

class UoinFind {
public:
	map<int, int> father;// 结点和其祖宗结点分别为key value
	map<int, int> size;// 一个集合的祖先结点和这个集合的大小分别为 key value
	//初始化的时候每个结点的祖先结点就是自己
	int findHead(int n) {
		stack<int> t;
		while (father[n] != n)
		{
			n = father[n];
			t.push(n);
		}
		while (!t.empty()) {
			father[t.top()] = n;
			t.pop();
		}
		return n;
	}

	void uoin(int a, int b) {
		int A = findHead(a);
		int B = findHead(b);
		if (A != B) {
			int maxSet = size[A] > size[B] ? A : B;
			int minSet = size[A] > size[B] ? B : A;
			father[minSet] = maxSet;
			size[maxSet] = size[maxSet] + size[minSet];
			size.erase(minSet);
		}
	}

	bool isSame(int a, int b) {
		return findHead(a) == findHead(b);
	}
};

struct Dlink {
	int key, value;
	Dlink* pre;
	Dlink* next;
	Dlink() :key(0), value(0), pre(nullptr), next(nullptr) {};
	Dlink(int _key, int _value) :key(_key), value(_value), pre(nullptr), next(nullptr) {};
};
class LRUcahe {
public:
	unordered_map<int, Dlink*>cahe;
	Dlink* head;
	Dlink* tail;
	int size;
	int capacity;

	LRUcahe(int _capacity) :capacity(_capacity), size(0) {
		head = new Dlink();
		tail = new Dlink();
		head->next = tail;
		tail->pre = head;
	}
	int get(int key, int value) {
		if (!cahe.count(key)) {
			Dlink* node = new Dlink(key, value);
			cahe[key] = node;
			addtohead(node);
			size++;
			if (size > capacity) {
				Dlink* removed = removeTail();
				cahe.erase(removed->key);
				delete removed;
				size--;
			}
			return -1;
		}
		Dlink* node = cahe[key];
		movetohead(node);
		return node->value;
	}
	void addtohead(Dlink* node) {
		node->pre = head;
		node->next = head->next;
		head->next->pre = node;
		head->next = node;
	}
	void removenode(Dlink* node) {
		node->pre->next = node->next;
		node->next->pre = node->pre;
	}
	void movetohead(Dlink* node) {
		removenode(node);
		addtohead(node);
	}
	Dlink* removeTail() {
		Dlink* node = tail->pre;
		removenode(node);
		return node;
	}
};

//第k大的数 建大顶堆
class Solution215 {
public:
	vector<int> heap;
	int len;
	//向下找
	void heapify(int index) {
		int left = index * 2 + 1;
		while (left < len) {
			int right = left + 1;
			int lagestIndex = INT_MIN;
			if (right < len) {
				lagestIndex = heap[index] > heap[left] ? index : left;
				lagestIndex = heap[lagestIndex] > heap[right] ? lagestIndex : right;
			}
			else lagestIndex = heap[index] > heap[left] ? index : left;
			if (lagestIndex == index)return;
			swap(heap[lagestIndex], heap[index]);
			index = lagestIndex;
			left = index * 2 + 1;
		}
	}
	int pop() {
		int res = heap[0];
		len--;
		swap(heap[0], heap[len]);
		heapify(0);
		return res;
	}
	//向上找
	void push(int val) {
		heap[len] = val;
		int index = len;
		while (heap[index] > heap[(index - 1) / 2]) {
			swap(heap[index], heap[(index - 1) / 2]);
			index = (index - 1) / 2;
		}
		len++;
	}
	void builheap(int size) {
		heap = vector<int>(size, 0);
		len = 0;
	}

	int findKthLargest(vector<int>& nums, int k) {
		builheap(nums.size());
		for (int i : nums)push(i);
		for (int i = 0; i < heap.size(); i++) {
			cout << heap[i] << endl;
		}
		int res = 0;
		for (int i = 0; i < k; i++) {
			res = pop();
		}
		return res;
	}
};
vector<int> partition(vector<int>& nums, int L, int R) {
	//R作为基准，返回排序后的等于 R 的左右边界 都是闭区间
	//less more 是两个指针 分别表示小于R的右边界，大于R的左边界 （ ]
	int less = L - 1;
	int more = R;
	//L表示当前位置
	while (L < more) {
		if (nums[L] < nums[R]) {
			less++;
			swap(nums[less], nums[L]);
			L++;
		}
		else if (nums[L] > nums[R]) {
			more--;
			swap(nums[more], nums[L]);
		}
		else L++;
	}
	swap(nums[R], nums[more]);
	return { less + 1,more };
}

int findMaxK(vector<int>& nums, int L, int R, int k) {
	if (L <= R) {
		int t = rand() % (R - L + 1) + L;
		swap(nums[t], nums[R]);
		vector<int> p = partition(nums, L, R);
		//如果等于nums[R]的数字只有一个
		if (p[0] == p[1]) {
			if (p[0] == nums.size() - k)return nums[p[0]];
			//如果要找的 k 在这个数字下标的左边 
			else if (p[0] > nums.size() - k)return findMaxK(nums, L, p[0] - 1, k);
			//如果要找的 k 在这个数字下标的右边
			else return findMaxK(nums, p[1] + 1, R, k);
		}
		//如果等于nums[R]的数字不止一个
		else if (p[0] == nums.size() - k)return nums[p[0]];
		else if (p[1] == nums.size() - k)return nums[p[1]];
		else {
			//如果要找的 k 在右边界下标的右边 
			if (p[1] < nums.size() - k) return findMaxK(nums, p[1] + 1, R, k);
			//如果要找的 k 在左边界下标的左边 
			else if (p[0] > nums.size() - k)return findMaxK(nums, L, p[0] - 1, k);
			//如果要找的 k 在左右边界中间 
			else return findMaxK(nums, p[0] + 1, p[1] - 1, k);
		}
	}
	return 0;
}

void quickSort(vector<int>& nums, int L, int R) {
	//防止越界 因为 partition 返回的 p 可能为 [0 0]
	if (L < R) {
		int t = rand() % (R - L + 1) + L;
		swap(nums[t], nums[R]);
		vector<int> p = partition(nums, L, R);
		quickSort(nums, L, p[0] - 1);
		quickSort(nums, p[1] + 1, R);
	}
}

class Lnode {
public:
	int val;
	Lnode* next;
	Lnode() :val(0), next(nullptr) {};
	Lnode(int v) {
		val = v;
		next = nullptr;
	}
};


Lnode* findK(Lnode* node, int k) {
	//从node开始返回第k个结点
	k--;
	while (k&&node) {
		k--;
		node = node->next;
	}
	return node;
}
void revers(Lnode* st, Lnode* ed) {
	//保存最后一个结点的下一个结点
	//例如 k = 3  1-->2-->3-->4
	//对1到3做完逆序后 1<--2<--3  4     3和4之间没有指向，如果不找一个指向4
	//4结点就无法被找到，无法进行后序操作
	ed = ed->next;
	Lnode* cur = st;
	Lnode* pre = nullptr;
	Lnode* next = nullptr;
	//对st和ed之间做逆序，做完后当前指针应该指向ed的后面
	while (cur != ed) {
		next = cur->next;
		cur->next = pre;
		pre = cur;
		cur = next;
	}
	//由于开始的结点再做完逆序后没有指向，让他指向最后一个结点的下一个结点
	st->next = ed;
}
Lnode* KtransLink(Lnode* head, int k) {
	//链表k分组逆序 每k个结点逆序 最后剩余不足k个的不管
	Lnode* start = head;
	Lnode* end = findK(head, k);
	if (end == nullptr)return head;
	//最终的头节点
	head = end;
	//做完逆序后没有指向，start指向最后一个结点的下一个结点
	revers(start, end);
	//前一段逆序后的最后一个结点
	Lnode* lastEnd = start;
	while (lastEnd->next) {
		start = lastEnd->next;
		end = findK(start, k);
		if (end == nullptr)return head;
		revers(start, end);
		lastEnd->next = end;
		lastEnd = start;
	}
	return head;

}

vector<int> SpiralMatrix(vector<vector<int> >& matrix) {
	//二维数组螺旋输出
	vector<int> ans;
	int m = matrix.size(), n = matrix[0].size();
	set<pair<int, int>> s;
	int c = 0, l = 0;
	int toal = m * n;
	while (toal) {
		while (s.find({ c,l }) == s.end()) {
			if (c >= m || l >= n)break;
			ans.push_back(matrix[c][l]);
			s.insert({ c,l });
			toal--;
			l++;
		}
		l--;
		c++;
		while (s.find({ c,l }) == s.end()) {
			if (l < 0 || c >= m)break;
			ans.push_back(matrix[c][l]);
			s.insert({ c,l });
			toal--;
			c++;
		}
		c--;
		l--;
		while (s.find({ c,l }) == s.end()) {
			if (c < 0 || l < 0)break;
			ans.push_back(matrix[c][l]);
			s.insert({ c,l });
			toal--;
			l--;
		}
		l++;
		c--;
		while (s.find({ c,l }) == s.end()) {
			if (l >= n || c < 0)break;
			ans.push_back(matrix[c][l]);
			s.insert({ c,l });
			toal--;
			c--;
		}
		c++;
		l++;
	}
	return ans;
}

int smoothWindow(vector<int>& arr, int k) {
	//滑动窗口
	int maxans = 0;
	int zero = 0;
	int l = 0, r = 0;
	for (r; r < arr.size(); r++) {
		if (arr[r] == 0)zero++;
		while (zero > k) {
			if (arr[l] == 0)zero--;
			l++;
		}
		maxans = max(maxans, r - l + 1);
	}
	return maxans;
}

int t1(vector<int>& nums, int k) {
	//有序数组nums，绳子长度K
	//问绳子最多可以压住几个点，包括边缘
	int l = 0, r = 0;
	int ans = 0;
	while (l < nums.size()) {
		while (r < nums.size() && nums[r] - nums[l] <= k) {
			r++;
		}
		ans = max(ans, r - l);
		l++;
	}
	return ans;
}


int t3(string& s) {
	//s只有B G两种字母，只可以相邻交换
	//问 B和G分别全部放在s的两边 最少需要交换多少次
	//如 BBBBBBGGGGGGG 或 GGGGGGBBBBBB
	int l = 0, step = 0;
	//修改为 GGGGGGBBBBBB
	for (int i = 0; i < s.size(); i++) {
		if (s[i] == 'G') {
			step = step + i - l;
			l++;
		}
	}
	int step2 = 0;
	l = 0;
	//修改为 BBBBBBGGGGGGG
	for (int i = 0; i < s.size(); i++) {
		if (s[i] == 'B') {
			step = step + i - l;
			l++;
		}
	}
	return min(step, step2);
}

int t4_dfs(vector<int>& nums, int ind, int sum, int tar) {
	if (ind >= nums.size()) {
		if (sum == tar)return 1;
		else return 0;
	}
	int p1 = t4_dfs(nums, ind + 1, sum + nums[ind], tar);
	int p2 = t4_dfs(nums, ind + 1, sum - nums[ind], tar);
	return p1 + p2;
}

int t4(vector<int>& nums, int tar) {
	return t4_dfs(nums, 0, 0, tar);
}

int t4_2(vector<int>& nums, int tar) {
	//P-N = T  -->  P-N + P+N = T + P+N   -->   2*P = T+SUM   -->  P = (T+SUM)/2 
	// T SUM 视为已知，问题转换为 再nums中选择一些数 使其和为 (T+SUM)/2 的方法数
	//dp[i][j] 在前i个元素中自由选择数据，使得和为 j 的方法数
	int sum = 0;
	for (int i : nums)sum = sum + i;
	if (sum < tar) return -1;
	if ((sum + tar) / 2 == 1)return -1;
	vector<vector<int>> dp(nums.size() + 1, vector<int>(1 + (sum + tar) / 2, 0));
	//dp 初始值
	dp[0][0] = 1;
	for (int i = 1; i <= dp.size(); i++) {
		for (int j = 0; j <= dp[0].size(); j++) {
			//                       不要nums[i]       要nums[i]
			if (j >= nums[i])dp[i][j] = dp[i - 1][j] + dp[i - 1][j - nums[i]];
		}
	}
	return dp[nums.size()].back();
}

int t5_dfs(vector<vector<int>>& value, int inde, int res) {
	//A 区域剩余名额为res情况下，把第Inde以后的司机(包括第inde)分给 A 区域，可以获得的最大钱数
	if (inde == value.size())return 0;
	//还剩余司机
	//刚好所有司机都去A
	if (res == value.size() - inde) {
		return value[inde][0] + t5_dfs(value, inde + 1, res - 1);
	}
	//A没有名额，司机只能去B
	if (res == 0) {
		return value[inde][1] + t5_dfs(value, inde + 1, res);
	}
	//司机可以去A B
	int p1 = value[inde][0] + t5_dfs(value, inde + 1, res - 1);
	int p2 = value[inde][1] + t5_dfs(value, inde + 1, res);
	return max(p1, p2);
}

int t5(vector<vector<int>>& value, int N) {
	//N个司机平均分给A B两地，已知每个司机在A B 的收益
	//问 使得所有司机总收益最高的钱数
	return t5_dfs(value, 0, N / 2);
}

int t7(vector<int>& num, int k) {
	//已知每个人的能力差别，且当两个人的能力差别为 k 的时候可以比赛
	//能同时比赛的最大场次
	sort(num.begin(), num.end());
	int l = 0, r = 0;
	int ans = 0;
	//同时比赛，每个人只能参加一个比赛
	vector<bool> used(num.size(), false);
	while (r < num.size() && l < num.size()) {
		if (used[l])l++;
		else if (l == r)r++;
		else if (num[r] - num[l] < k)r++;
		else if (num[r] - num[l] > k)l++;
		else if (num[r] - num[l] == k) {
			used[r] = true;
			used[l] = true;
			ans++;
			l++;
			r++;
		}
	}
	return ans;
}

void t7_dfs(vector<int>& num, vector<vector<int>>& res, vector<int>& path, vector<bool>& used) {
	//全排列
	if (path.size() == num.size()) {
		res.push_back(path);
		return;
	}
	for (int i = 0; i < num.size(); i++) {
		if (used[i] == false) {
			used[i] = true;
			path.push_back(num[i]);
			t7_dfs(num, res, path, used);
			path.pop_back();
			used[i] = false;
		}
	}
	return;
}

int t8(vector<int>& num, int limit) {
	//每个人的体重，船的限重（每个船相同）
	//num[i]<=limit
	sort(num.begin(), num.end());
	//贪心
	//找第一个大于Limit/2的位置
	int r = upper_bound(num.begin(), num.end(), limit / 2) - num.begin();
	int index = r;
	int l = r - 1;
	int len = num.size();
	vector<int> flg(len, 0);
	while (l >= 0 && r < len) {
		while (l >= 0 && num[r] + num[l] > limit) {
			l--;
		}
		while (l >= 0 && r < len&&num[l] + num[r] <= limit) {
			flg[r] = 1;
			flg[l] = 1;
			l--;
			r++;
		}
	}
	int p0 = 0, p1 = 0, p2 = 0;
	//p0 右边的剩余人数
	//p1 左右两边各一个人 可以两个人坐一个船的人数
	//p2 左边剩余的人数
	for (int i = 0; i < len; i++) {
		if (i >= index && flg[i] == 0)p0++;
		if (flg[i] == 1)p1++;
		if (i < index&&flg[i] == 0)p2++;
	}
	return p0 + p1 / 2 + (int)ceil(p2 / 2);
}

int t9(vector<int>& num) {
	//子数组的最大累加和
	//子数组，以第i个位置的数字为结尾的子数组的累计和
	int ans = 0;
	vector<int> dp(num.size(), 0);
	dp[0] = num[0];
	ans = dp[0];
	for (int i = 1; i < num.size(); i++) {
		//只要当前的一个   以num[i]结尾的
		dp[i] = fmax(num[i], num[i] + dp[i - 1]);
		ans = max(dp[i], ans);
	}
	return ans;
}

int t10(vector<int>& nums) {
	//n 个孩子站成一排。给你一个整数数组 ratings 表示每个孩子的评分。
	//你需要按照以下要求，给这些孩子分发糖果：
	//每个孩子至少分配到 1 个糖果。
	//相邻两个孩子评分更高的孩子会获得更多的糖果。
	//请你给每个孩子分发糖果，计算并返回需要准备的 最少糖果数目 。
	vector<int> dp(nums.size());
	vector<int> dp2(nums.size());
	dp[0] = 1, dp2[nums.size() - 1] = 1;
	int ans = 0;
	for (int i = 1; i < nums.size(); i++) {
		if (nums[i] > nums[i - 1])dp[i] = dp[i - 1] + 1;
		else dp[i] = 1;
	}
	for (int i = nums.size() - 1; i > 0; i--) {
		if (nums[i - 1] > nums[i])dp2[i - 1] = dp2[i] + 1;
		else dp2[i - 1] = 1;
	}
	for (int i = 0; i < nums.size(); i++) {

		ans = ans + max(dp[i], dp2[i]);
	}
	return ans;
}

int t11(string str1, string str2, string str3) {
	//给定三个字符串 s1、s2、s3，请判断 s3 能不能由 s1 和 s2 交织（交错） 组成。
	int len1 = str1.size(), len2 = str2.size(), len3 = str3.size();
	if (len1 + len2 != len3)return false;
	vector<vector<bool>> dp(len1 + 1, vector<bool>(len2 + 1));
	//dp[i][j] s1 的前 i 个字符和 s2 的前 j 个字符能否组成 s3 的前i+j
	dp[0][0] = true;
	for (int i = 1; i <= len1; i++) {
		if (str1[i - 1] == str3[i - 1])dp[i][0] = true;
		else break;
	}
	for (int i = 1; i <= len2; i++) {
		if (str2[i - 1] == str3[i - 1])dp[0][i] = true;
		else break;
	}
	for (int i = 1; i <= len1; i++) {
		for (int j = 1; j <= len2; j++) {
			//s1 的前 i 个字符和 s2 的前 j 个字符要组成 s3 的前i+j字符两种可能
			//1  s1 的第 i 个字符和 s3 的第i+j个字符相等 且
			//   s1 的前 i-1 个字符和 s2 的前 j 个字符组成了 s3 的前i+j-1个字符
			//2  s2 的第 i 个字符和 s3 的第i+j个字符相等 且
			//   s1 的前 i 个字符和 s2 的前 j-1 个字符组成了 s3 的前i+j-1个字符
			if (str1[i - 1] == str3[i + j - 1] && dp[i - 1][j] || str2[j - 1] == str3[i + j - 1] && dp[j][i - 1])
				dp[i][j] = true;

		}
	}
	return dp[len1][len2];
}


bool t14_isSame(Tnode* h1, Tnode* h2) {
	if ((h1 == nullptr&&h2 != nullptr) || (h2 == nullptr&&h1 != nullptr))return false;
	if (h1 == nullptr&&h2 == nullptr)return true;
	return h1->val == h2->val&&t14_isSame(h1->left, h2->left) && t14_isSame(h1->right, h2->right);

}

int t14(Tnode* root) {
	//求一棵二叉树上相等子树的数量
	if (root == nullptr)return 0;
	//左边树上的相等子树数量
	int p1 = t14(root->left);
	//右边树上的相等子树数量
	int p2 = t14(root->right);
	//左右子树是否相等
	int p3 = t14_isSame(root->left, root->right) == true ? 1 : 0;
	return p1 + p2 + p3;
}

//编辑距离//
int t15(string t1, string t2, int add, int rep, int del) {
	//编辑距离问题   返回把t1修改为t2的最小代价
	//添加字母 代价add  替换字母  代价rep   删除字母  代价del
	int len1 = t1.size(), len2 = t2.size();
	//dp[i][j] 把t1的前i个字符编辑为t2的前j个字符 需要的最小代价
	vector<vector<int>> dp(len1 + 1, vector<int>(len2 + 1));
	dp[0][0] = 0;
	//把t1的前0个字符编辑为t2的前i个字符 需要的最小代价
	for (int i = 1; i <= len1; i++) {
		dp[0][i] = add * i;
	}
	//把t1的前i个字符编辑为t2的前0个字符 需要的最小代价
	for (int i = 1; i <= len2; i++) {
		dp[i][0] = del * i;
	}
	for (int i = 1; i <= len1; i++) {
		for (int j = 1; j <= len2; j++) {
			if (t1[i - 1] == t2[j - 1])dp[i][j] = dp[i - 1][j - 1];
			if (t1[i - 1] != t2[j - 1])dp[i][j] = dp[i - 1][j - 1] + rep;
			//t1的前i-1个字母在添加一个字母成为t2的前j个字母
			dp[i][j] = min(dp[i][j], dp[i - 1][j] + add);
			//t1的前i个字母在删除一个字母成为t2的前j-1个字母
			dp[i][j] = min(dp[i][j], dp[i][j - 1] + del);
		}
	}
	return dp[len1][len2];
}

int t16(string t1, string t2) {
	//字符串t1的子序列中出现字符串t2的次数
	//问题转换：从字符串t1中选择字符，有多少种方法可以使得最终的选择结果为t2
	int len1 = t1.size(), len2 = t2.size();
	//dp[i][j] 从t1的前 i 个字符中选择，有多少种方法可以使得最终的选择结果为t2的前 j 个字符
	vector<vector<int>> dp(len1 + 1, vector<int>(len2 + 1));
	dp[0][0] = 1;
	for (int i = 1; i <= len1; i++) {
		dp[i][0] = 1;
	}
	for (int i = 0; i <= len2; i++) {
		dp[0][i] = 0;
	}
	for (int i = 1; i <= len1; i++) {
		for (int j = 1; j <= len2; j++) {
			if (t1[i - 1] == t2[j - 1]) {
				//从t1的前i-1个字符中选择，可以组成t2的前j-1字符的方法 
				// + 从t1的前i-1个字符中选择，可以组成t2的前j字符的方法 
				//实际上是：是否用t1[i-1]来匹配的问题
				dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
			}
			else dp[i][j] = dp[i - 1][j];
		}
	}
	return dp[len1][len2];

}

int t17(vector<int>& num) {
	//挡板内容纳的最大水量
	int l = 0, r = num.size() - 1;
	int ans = 0;
	int res = 0;
	while (l < r) {
		res = min(num[l], num[r])*(r - l);
		if (num[l] < num[r])l++;
		else r--;
		ans = max(res, ans);
	}
	return ans;
}


//跳跃游戏2
int t19(vector<int>& nums) {
	//nums[i] 代表在 i 位置时可以跳的距离
	//返回到达最后一个位置需要的步数
	int step = 0;
	//cur 如果不增加步数 最远能到达的位置
	int cur = 0;
	//next 下一跳可以到达的最远位置
	int next = nums[0];
	for (int i = 0; i < nums.size(); i++) {
		if (i > cur) {
			step++;
			cur = next;
			//下一步能到达的位置 先置 0 
			next = 0;
		}
		next = max(next, nums[i] + i);
	}
	return step;
}

//返回 str[l,r]上为 true 的方法数 和为false的方法数
vector<int> t20_dfs(string& str, int l, int r) {
	int t = 0, f = 0;
	if (l == r) {
		if (str[l] == '1')t++;
		if (str[l] == '0')f++;
	}
	else {
		//l---r 之间大于3
		// i 指向的是每一个逻辑符号，对每个逻辑符号都尝试作为最后一个运算符
		for (int i = l + 1; i < r; i = i + 2) {
			vector<int> Left = t20_dfs(str, l, i - 1);
			vector<int> Right = t20_dfs(str, i + 1, r);
			int Lt = Left[0], Lf = Left[1];
			int Rt = Right[0], Rf = Right[1];
			if (str[i] == '&') {
				t = t + Lt * Rt;
				f = f + Lt * Rf + Lf * Rf + Lf * Rt;
			}
			if (str[i] == '|') {
				t = t + Lt * Rt + Lt * Rf + Lf * Rt;
				f = f + Lf * Rf;
			}
			if (str[i] == '^') {
				t = t + Lt * Rf + Lf * Rt;
				f = f + Lt * Rt + Lf * Rf;
			}
		}
	}
	return { t,f };
}

//记忆化搜索
vector<int> t20__dfs(string& str, int l, int r, vector<vector<vector<int>>>& dp) {
	//命中直接返回
	if (dp[l][r][0] != dp[l][r][1])return dp[l][r];
	int t = 0, f = 0;
	if (l == r) {
		if (str[l] == '1')t++;
		if (str[l] == '0')f++;
	}
	else {
		//l---r 之间大于3
		// i 指向的是每一个逻辑符号，对每个逻辑符号都尝试作为最后一个运算符
		for (int i = l + 1; i < r; i = i + 2) {
			vector<int> Left = t20__dfs(str, l, i - 1, dp);
			vector<int> Right = t20__dfs(str, i + 1, r, dp);
			int Lt = Left[0], Lf = Left[1];
			int Rt = Right[0], Rf = Right[1];
			if (str[i] == '&') {
				t = t + Lt * Rt;
				f = f + Lt * Rf + Lf * Rf + Lf * Rt;
			}
			if (str[i] == '|') {
				t = t + Lt * Rt + Lt * Rf + Lf * Rt;
				f = f + Lf * Rf;
			}
			if (str[i] == '^') {
				t = t + Lt * Rf + Lf * Rt;
				f = f + Lt * Rt + Lf * Rf;
			}
		}
	}
	//没有命中，返回前存储
	dp[l][r] = { t,f };
	return { t,f };
}

//布尔表达式的期待表达数
//给定一个布尔表达式和一个期望的布尔结果 result，
//布尔表达式由 0 (false)、1 (true)、& (AND)、 | (OR) 和 ^ (XOR) 符号组成。
//实现一个函数，算出有几种可使该表达式得出 result 值的括号方法。

//问题转化 对每个逻辑符号都尝试作为最后一个运算符，得到总的方法数
int t20(string& str, int result) {
	vector<vector<vector<int>>> dp(str.size(), vector<vector<int>>(str.size(), vector<int>(2)));
	vector<int> p = t20__dfs(str, 0, str.size() - 1, dp);
	return result == 1 ? p[0] : p[1];
}

/*
	从面值为1--10的牌等概率的中有放回的抽牌
	累计和小于 17 继续抽牌
	累加和 >=17 且 <21 时获胜
	累加和 >=21 时失败
	返回获胜的概率
*/
//累加和为 sum 时的获胜概率
double t21_dfs(int sum) {
	if (sum >= 17 && sum < 21)return 1.0;
	if (sum >= 21)return 0.0;
	double p = 0.0;
	for (int i = 1; i <= 10; i++) {
		p = p + t21_dfs(sum + i);
	}
	return p / 10;
}
double t21() {
	return t21_dfs(0);
}

/*
	约瑟夫环问题
	人数改为N，报到M时，把那个人杀掉，那么数组是怎么移动的？
	每杀掉一个人，下一个人成为头，相当于把数组向前移动M位。
	若已知N-1个人时，胜利者的下标位置位f(N ?? 1 , M) 则N个人的时候，就是往后移动M位，
	(因为有可能数组越界，超过的部分会被接到头上，所以还要模N)，
	既f(N , M ) = (f(N ?? 1 , M) + M) % n
	求出的结果是数组下标，最后结果编号还要+1

*/
int t22(int N, int M) {
	if (N == 1)return N;
	return (t22(N - 1, M) + M) % N;
}

/*
假设有 n??台超级洗衣机放在同一排上。开始的时候，每台洗衣机内可能有一定量的衣服，也可能是空的。
在每一步操作中，你可以选择任意 m (1 <= m <= n) 台洗衣机，与此同时将每台洗衣机的一件衣服送到相邻的一台洗衣机。
给定一个整数数组??machines 代表从左至右每台洗衣机中的衣物数量，
请给出能让所有洗衣机中剩下的衣物的数量相等的 最少的操作步数
*/
int t23(vector<int>& nums) {
	//针对每个点，讨论其左右两侧达到平均需要的轮数
	int sum = 0;
	for (int i = 0; i < nums.size(); i++) {
		sum = sum + nums[i];
	}
	if (sum%nums.size() != 0)return -1;
	int ans = 0;
	int leftSum = 0;
	int arvg = sum / nums.size();
	for (int i = 0; i < nums.size(); i++) {
		//左侧衣服数量与平均之间的距离
		int leftRes = leftSum - i * arvg;
		//右侧衣服数量与平均之间的距离
		//               右侧拥有的衣服             右侧期待的平均衣服
		int rightRes = sum - leftSum - nums[i] - (nums.size() - i - 1)*arvg;
		if (leftRes < 0 && rightRes < 0) {
			ans = max(ans, abs(leftRes) + abs(rightRes));
		}
		else {
			ans = max(ans, max(abs(leftRes), abs(rightRes)));
		}
		leftSum = leftSum + nums[i];
	}
	return ans;
}


//字符串s1中有多少子序列等于字符串s2
int t25(string s1, string s2) {
	int len1 = s1.size(), len2 = s2.size();
	vector<vector<int>> dp(len1 + 1, vector<int>(len2 + 1));
	//dp[i][j] s1的前i个字符的子序列中有多少等于s2的前j个字符串
	dp[0][0] = 1;
	for (int j = 1; j <= len2; j++)dp[0][j] = 0;
	for (int i = 1; i < len1; i++)dp[i][0] = 1;
	for (int i = 1; i <= len1; i++) {
		for (int j = 1; j < len2; j++) {
			dp[i][j] = dp[i - 1][j];
			if (s1[i - 1] == s2[j - 1])
				dp[i][j] = dp[i][j] + dp[i - 1][j - 1];
		}
	}
	return dp[len1][len2];
}

//字符串s中有多少不同的子序列
//抓住 以某个字符结尾的子序列的个数
int t26(string s) {
	vector<int> count(26, 0);
	int all = 1;
	int curent = 0;
	for (int i = 0; i < s.size(); i++) {
		//求模 两个数相减后取模
		//因为相减后不确定正负，所以加上一个模数后再取模
		curent = (all - count[s[i] - 'a'] + 1000000007) % 1000000007;
		all = (curent + all) % 1000000007;
		count[s[i] - 'a'] = (count[s[i] - 'a'] + curent) % 1000000007;
	}
	return (all - 1);
}

/*
关于二分  伪代码
l = -1,r = N;
while((l+1)!=r){
	mid = l + (r-l)/2;
	if isblue(m) l = mid;
	else r = mid;
}
return l or r;

例子 1 2 3 5 5 5 8 9
					   条件   return
第一个 >= 5   的元素     <5       r         lower_bound()
最后一个 < 5  的元素     <5       l
第一个 > 5    的元素     <=5      r         upper_bound()
最后一个 <= 5 的元素     <=5      l
*/

void t27_dfs(vector<vector<int>>& grid, int m, int n, int i, int j) {
	if (i >= 0 && i < m&&j >= 0 && j < n) {
		cout << i << " " << j << endl;
		grid[i][j] = 2;
		if (i - 1 >= 0 && grid[i - 1][j] == 1)t27_dfs(grid, m, n, i - 1, j);
		if (j - 1 >= 0 && grid[i][j - 1] == 1)t27_dfs(grid, m, n, i, j - 1);
		if (i + 1 < m&&grid[i + 1][j] == 1)t27_dfs(grid, m, n, i + 1, j);
		if (j + 1 < n&&grid[i][j + 1] == 1)t27_dfs(grid, m, n, i, j + 1);
	}
	else return;
}
//最短的桥
int t27(vector<vector<int>>& grid) {
	//dfs找到第一块陆地
	bool findOne = false;
	int m = grid.size(), n = grid[0].size();
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (!findOne&&grid[i][j] == 1) {
				t27_dfs(grid, m, n, i, j);
				findOne = true;
			}
		}
	}
	//bfs 找另一块陆地
	queue<vector<int>> q;
	//把第一块大陆放进队列
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (grid[i][j] == 2)q.push({ i,j });
		}
	}
	int step = 0;
	while (!q.empty()) {
		int len = q.size();
		for (int i = 0; i < len; i++) {
			vector<int> cur = q.front();
			q.pop();
			int x = cur[0], y = cur[1];
			if ((x + 1 < m&&grid[x + 1][y] == 1) || (y + 1 < n&&grid[x][y + 1] == 1)
				|| (x - 1 >= 0 && grid[x - 1][y] == 1) || y - 1 >= 0 && grid[x][y - 1] == 1) {
				return step;
			}
			if (x + 1 < m&&grid[x + 1][y] == 0) {
				grid[x + 1][y] = 2;
				q.push({ x + 1,y });
			}
			if (y + 1 < n&&grid[x][y + 1] == 0) {
				grid[x][y + 1] = 2;
				q.push({ x,y + 1 });
			}
			if (x - 1 >= 0 && grid[x - 1][y] == 0) {
				grid[x - 1][y] = 2;
				q.push({ x - 1,y });
			}
			if (y - 1 >= 0 && grid[x][y - 1] == 0) {
				grid[x][y - 1] = 2;
				q.push({ x,y - 1 });
			}
		}
		step++;
	}
	return step;
}

struct t28cmp {
	bool operator()(const vector<int>&a, const vector<int>&b) {
		return (a[0] > b[0]);
	}
};
//最小区间 leetcode 632 set 插入 vector （自定义排序）出错  VScode 结果正确 
vector<int> t28(vector<vector<int>>& nums) {
	/*map使用
	map<int,vector<int>> m;
	插入
	m.insert(pair<int,vector<int>>(nums[i][0],{i,0}));
	//m 中放的是 数字 和 位置(行 列)
	//访问第一个和最后一个元素
	map<int,vector<int>>::iterator it = m.end();
	int num_min = m.begin()->first;
	vector<int> num_min_position = m.begin()->second;
	it--;
	int num_max = it->first;
	vector<int> num_max_position = it->second;
	//删除
	m.erase(m.begin());
	//判断插入是否成功
	auto ret = m.insert(pair<int,vector<int>>(nums[next_ele_col][next_ele_row],{next_ele_col,next_ele_row}));
	if(!ret.second&&m.size()==1)return {m.begin()->first,m.begin()->first};
	*/
	//map中的key不可以重复，优先队列可以有重复的值
	//小顶堆
	//priority_queue<int,vector<int>,greater<int>> p;

	//重载（）运算符 根据向量中的第一个元素从大到小排序

	//存放 ： 数字 行 列
	//先把每行的第一个数放进去
	set<vector<int>, t28cmp> s;
	for (int i = 0; i < nums.size(); i++) {
		s.insert({ nums[i][0],i,0 });
	}
	set<vector<int>>::iterator it;
	int res = INT_MAX;
	vector<int> ans;
	//直到 某一行没有下一个元素了 就结束
	//set里的最大值和最小值的差更新结果 
	//删除最小值 添加删除的值所在行的下一个数
	while (1) {
		vector<int> max = *s.begin();
		it = s.end();
		it--;
		vector<int> min = *it;
		if (max[0] - min[0] < res) {
			res = max[0] - min[0];
			ans = { max[0],min[0] };
		}
		if (min[2] + 1 >= nums[min[1]].size())return ans;
		s.erase(it);
		s.insert({ nums[min[1]][min[2] + 1],min[1],min[2] + 1 });
	}
	return ans;

}

//接雨水
int t29(vector<int>& nums) {
	if (nums.size() <= 2)return 0;
	//双指针 对于每一个点上计算其可以留住的最大水量 累加
	//左指针所指处 可以留的最大水量由左指针左侧的最大值和其本身的值共同决定
	//右指针所指处 可以留的最大水量由右指针右侧的最大值和其本身的值共同决定   
	int left = 1, right = nums.size() - 2;
	int left_max = nums[0], right_max = nums.back();
	int ans = 0;
	while (left <= right) {
		if (left_max < right_max) {
			ans = ans + max(left_max - nums[left], 0);
			left_max = max(left_max, nums[left]);
			left++;
		}
		else {
			ans = ans + max(right_max - nums[right], 0);
			right_max = max(right_max, nums[right]);
			right--;
		}
	}
	return ans;
}

struct t30cmp
{//按照第一个元素 升序
	bool operator()(vector<int>& a, vector<int>& b) {
		return a[0] > b[0];
	}
};

//二维接雨水
int t30(vector<vector<int>>& nums) {
	int m = nums.size(), n = nums[0].size();
	vector<vector<int>> used(m, vector<int>(n, 0));
	int ans = 0;
	int MAX = 0;
	//只保证堆顶是最小值 不保证其他顺序
	priority_queue<vector<int>, vector<vector<int>>, t30cmp> p;
	//矩阵第一圈进入队列 ：数字 行 列
	//第一行
	for (int i = 0; i < n; i++) {
		p.push({ nums[0][i],0,i });
		used[0][i] = 1;
	}
	//最后一行
	for (int i = 0; i < n; i++) {
		p.push({ nums[m - 1][i],m - 1,i });
		used[m - 1][i] = 1;
	}
	//第一列
	for (int i = 1; i < m - 1; i++) {
		p.push({ nums[i][0],i,0 });
		used[i][0] = 1;
	}
	//最后一列
	for (int i = 1; i < m - 1; i++) {
		p.push({ nums[i][n - 1],i,n - 1 });
		used[i][n - 1] = 1;
	}
	while (!p.empty()) {
		//拿出元素时更新最大值
		vector<int> cur = p.top();
		p.pop();
		MAX = max(MAX, cur[0]);
		//把每一个拿出来的元素的上下左右得到元素放进去 
		//放的时候计算水量
		if (cur[1] + 1 < m&&used[cur[1] + 1][cur[2]] == 0) {
			ans = ans + max(MAX - nums[cur[1] + 1][cur[2]], 0);
			p.push({ nums[cur[1] + 1][cur[2]],cur[1] + 1,cur[2] });
			used[cur[1] + 1][cur[2]] = 1;
		}
		if (cur[1] - 1 >= 0 && used[cur[1] - 1][cur[2]] == 0) {
			ans = ans + max(MAX - nums[cur[1] - 1][cur[2]], 0);
			p.push({ nums[cur[1] - 1][cur[2]],cur[1] - 1,cur[2] });
			used[cur[1] - 1][cur[2]] = 1;
		}

		if (cur[2] + 1 < n&&used[cur[1]][cur[2] + 1] == 0) {
			ans = ans + max(MAX - nums[cur[1]][cur[2] + 1], 0);
			p.push({ nums[cur[1]][cur[2] + 1],cur[1],cur[2] + 1 });
			used[cur[1]][cur[2] + 1] = 1;
		}
		if (cur[2] - 1 >= 0 && used[cur[1]][cur[2] - 1] == 0) {
			ans = ans + max(MAX - nums[cur[1]][cur[2] - 1], 0);
			p.push({ nums[cur[1]][cur[2] - 1],cur[1],cur[2] - 1 });
			used[cur[1]][cur[2] - 1] = 1;
		}
	}
	return ans;
}


//在[l,r]上分为t堆的最小代价 每k个可以合并一堆
int t32_dfs(int l, int r, int t, vector<int>& nums, int k, vector<int>& pre) {
	if (r == l) {
		//如果 只有一个数 且要求分为1堆的时候 返回0 
		//返回0 后序根据前缀和计算代价
		return t == 1 ? 0 : -1;
	}
	if (t == 1) {
		int next = t32_dfs(l, r, k, nums, k, pre);
		if (next == -1)return -1;
		else return next + pre[r + 1] - pre[l];
	}
	else {
		int ans = INT_MAX;
		for (int i = l; i < r; i = i + k - 1) {
			//i做中间点 左边合并1堆 右边合并t-1堆
			int p1 = t32_dfs(l, i, 1, nums, k, pre);
			int p2 = t32_dfs(i + 1, r, t - 1, nums, k, pre);
			//如果有一边合并失败，则这个中间点无效
			if (p1 == -1 || p2 == -1) {
				return -1;
			}
			else ans = min(ans, p1 + p2);
		}
		return ans;
	}
}
//相邻k个石子可以合并，返回合并成1堆的最小代价
int t32(vector<int>& nums, int k) {
	vector<int> pre(nums.size() + 1);
	//计算前缀和 递归中使用 
	for (int i = 0; i < nums.size(); i++) {
		pre[i + 1] = pre[i] + nums[i];
	}
	return t32_dfs(0, nums.size() - 1, 1, nums, k, pre);
}

//最小包含子串 滑动窗口
//str1中包含所有str2的字符的最短连续子串得到长度 
//包含时可以不连续 可以有多余
string t33(string& str1, string& str2) {
	if (str1.size() < str2.size())return "";
	//记录str2的所有字符
	map<char, int> m;
	for (char c : str2) {
		m[c]++;
	}
	int all = str2.size();
	int l = 0, r = 0;
	int ans = INT_MAX;
	string res;
	while (r < str1.size()) {
		while (r < str1.size() && all>0)
		{
			//如果str2中存在当前字符 
			if (m.count(str1[r])) {
				//str2中当前对应字符的数量减一
				m[str1[r]]--;
				//如果减完后的值大于等于0 这是有效的减一 all要减一
				if (m[str1[r]] >= 0)all--;
			}
			//all等于0的时候 找到了一个包含str1的字符串子串
			//右侧窗口停止扩展
			if (all == 0)break;
			r++;
		}
		//只要左侧窗口缩减后all还为0 就说明目前的窗口还满足 
		//继续缩小窗口
		while (all == 0) {
			//记录当前答案
			if (r - l + 1 < ans) {
				ans = r - l + 1;
				res = str1.substr(l, ans);
			}
			//如果str2中存在当前字符 
			if (m.count(str1[l])) {
				//str2中当前对应字符的数量加一
				m[str1[l]]++;
				//如果加完后的值大于0 这时候窗口中就不满足包含str2了 
				//all加一
				if (m[str1[l]] > 0)all++;
			}
			l++;
		}
		//左侧缩到不满足的时候 右侧需要重新向右扩展
		r++;

	}
	return res;
}

//去除重复字母 使结果字典序最小
string t34(string& str) {
	if (str.size() < 2)return str;
	map<char, int> m;
	for (char c : str)m[c]++;
	int i;
	for (i = 0; i < str.size(); i++) {
		m[str[i]]--;
		if (m[str[i]] == 0)break;
	}
	//str[i]之后的字符（不包括str[i]）不能满足所有的字符都存在其中

	//找到前一段中的字典序最小的字符，如果有多个最小的 找第一个
	//把前一段中这个字符以前的字符全部删掉，把后一段中这个字符全部删除
	//实际上 是  把它（前一段中的字典序最小的字符）和它后面的不等于它的字符组成新的字符串
	//然后递归
	char key = 123;//最大的字符
	int min_key_indedx = 0;
	for (int j = 0; j <= i; j++) {
		//如果有多个最小的 找第一个
		if (str[j] < key) {
			key = str[j];
			min_key_indedx = j;
		}
	}
	string tmp1;
	for (int k = min_key_indedx + 1; k < str.size(); k++) {
		if (str[k] != key)tmp1.push_back(str[k]);
	}
	string tmp3;
	tmp3.push_back(key);
	return tmp3 + t34(tmp1);
}

//文化衫问题 返回最少的人数
//nums[i] 表示人数 除了i之外的还穿着和i一样的衬衫的人数
int t35(vector<int>&nums) {
	//eg [1,1,1,1,2,2,3,3,3,4]
	map<int, int> m;
	for (int i : nums)m[i]++;
	int ans = 0;
	for (map<int, int>::iterator it = m.begin(); it != m.end(); it++) {
		int tmp = it->second / (it->first + 1) + it->second % (it->first + 1) == 0 ? 0 : 1;
		ans = ans + tmp * (it->first + 1);
	}
	return ans;
}

//有效的括号
bool t36(string& str) {
	stack<char> s;
	for (char c : str) {
		if (c == '(' || c == '[' || c == '{') {
			if (c == '(')s.push(')');
			if (c == '[')s.push(']');
			if (c == '{')s.push('}');
		}
		else if (s.empty())return false;
		else {
			if (s.top() != c)return false;
			s.pop();
		}
	}
	return s.empty();
}

//填数独
void t38_dfs(vector<vector<char>>& board, vector<vector<bool>>& row, vector<vector<bool>>& col,
	vector<vector<bool>>& buck, int m, int n, bool vail) {
	//最后一个变量 用来判断是否正常填完，如果是因为出错则需要返回 回溯
	//如果正常填完 则返回 不用回溯
	if (m > 8) {
		//填完 返回
		vail = true;
		return;
	}
	//根据当前的行列 m n 确定下一个点的行列 nexti nextj
	//主要考虑是否要换行
	int nexti = n == 8 ? (m + 1) : m;
	int nextj = n == 8 ? 0 : n + 1;
	//如果当前位置已经填过了 直接到下一个点
	if (board[m][n] != '.')t38_dfs(board, row, col, buck, nexti, nextj, vail);
	//如果当前位置没有填 则一一尝试
	else {
		int b = 3 * (m / 3) + (n / 3);
		for (int k = 1; k <= 9; k++) {
			//如果这个数字在这一行 列 9宫格都没出现过 才可以尝试
			if ((!row[m][k]) && (!col[n][k]) && (!buck[b][k])) {
				row[m][k] = true;
				col[n][k] = true;
				buck[b][k] = true;
				board[m][n] = k + '0';
				t38_dfs(board, row, col, buck, nexti, nextj, vail);
				//如果是填完后返回的 则不在回溯
				//如果是填错了返回的 需要回溯
				if (!vail) {
					row[m][k] = false;
					col[n][k] = false;
					buck[b][k] = false;
					board[m][n] = '.';
				}
			}
		}
		//填错 返回
		return;
	}
}
void t38(vector<vector<char>>& board) {
	//第i行的j有没有出现过 i:[0 8] j:[1 9]
	vector<vector<bool>> row(9, vector<bool>(10));
	//第i列的j有没有出现过
	vector<vector<bool>> col(9, vector<bool>(10));
	//第i个方块的j有没有出现过
	vector<vector<bool>> buck(9, vector<bool>(10));
	for (int i = 0; i < 9; i++) {
		for (int j = 0; j < 9; j++) {
			if (board[i][j] == '.')continue;
			int tmp = 3 * (j / 3) + i / 3;
			int num = board[i][j] - '0';
			row[i][num] = true;
			col[j][num] = true;
			buck[tmp][num] = true;
		}
	}
	t38_dfs(board, row, col, buck, 0, 0, false);
}

//求多次方
double t39(double x, int n) {
	if (n == 0)return 1.0;
	int tmp = abs(n);
	double ans = x;
	double res = x;
	//把 n 用二进制表示 根据这个二进制数的每一个位置上是 0 还是 1 
	//判断要不要把x的这个二进制位代表数的次方乘上去
	//
	while (tmp) {
		if (tmp & 1)ans = ans * res;
		tmp = tmp >> 1;
		res = res * res;
	}
	return n > 0 ? ans : 1 / ans;
}

//开方 
//0到x上的二分 找第一个大于等sqrt(x)的值
int t40(int x) {
	int l = 0;
	int r = x + 1;
	while ((l + 1) != r) {
		int mid = l + (r - l) / 2;
		if (mid*mid <= x)l = mid;
		else r = mid;
	}
	return l;
}

//矩阵置0
void t41(vector<vector<int>>& matrix) {
	//记录有0的行 列
	vector<bool> row(matrix.size());
	vector<bool> col(matrix[0].size());
	for (int i = 0; i < matrix.size(); i++) {
		for (int j = 0; j < matrix[0].size(); j++) {
			if (matrix[i][j] == 0) {
				row[i] = true;
				col[j] = true;
			}
		}
	}
	//有0的行 全部置0
	for (int i = 0; i < row.size(); i++) {
		if (row[i] == true) {
			for (int j = 0; j < matrix[0].size(); j++) {
				matrix[i][j] = 0;
			}
		}
	}
	//有0的列 全部置0
	for (int i = 0; i < col.size(); i++) {
		if (col[i] == true) {
			for (int j = 0; j < matrix.size(); j++) {
				matrix[j][i] = 0;
			}
		}
	}
	return;
}

//乘积最大子数组
//两个数组 nums[i]结尾的子数组的乘积的最大和最小值
int t42(vector<int>& nums) {
	vector<int> dpmin(nums.size());
	vector<int> dpmax(nums.size());
	int res = nums[0];
	dpmin[0] = nums[0];
	dpmax[0] = nums[0];
	for (int i = 1; i < nums.size(); i++) {
		dpmax[i] = max(nums[i], max(dpmax[i - 1] * nums[i], dpmin[i - 1] * nums[i]));
		dpmin[i] = min(nums[i], min(dpmax[i - 1] * nums[i], dpmin[i - 1] * nums[i]));
		res = max(res, dpmax[i]);
	}
	return res;
}


//出卷子的方法数
int t46(vector<int>&nums, int m) {
	//已知每个题目得到难度值 载数组中
	//要求 前一个题的难度值不能超过后一题的难度值+m
	//返回满足要求的出题顺序
	//先排序 然后遍历 遍历到每个数的时候 计算这个数以前有多少数大于等于 nums[i]-m
	//把nums[i]往前面插，有多少个数大于等于 nums[i]-m ，就有多少种插法
	/*对于前面的一个数 x     如果 x>nums[i]-m*/
	/*则 nums[i]<x+m  所以 当nums[i] 插到x前面的时候是符合出题的难度值要求的*/
	//对于之前的每一种方法数，都可以有同样多的插法 所以 res*count
	//再加上 nums[i]不往前面插，也就是nums[i]之前的方法数

	//lower_bound() 第一个大于等于的数字的迭代器
	//upper_bound() 第一个大于的数字的迭代器
	int res = 1;//遍历nums[i]之前得到的方法数
	sort(nums.begin(), nums.end());
	//每遍历到一个数，更新一下方法数
	for (int i = 1; i < nums.size(); i++) {
		int count = (nums.begin() + i) - lower_bound(nums.begin(), nums.begin() + i, nums[i] - m);
		//遍历到nums[i]时得到的方法数 
		int cur = res + res * count;
		res = cur;
	}
	return res;
}

//查询人名
bool t48_h(int i, int j) {
	return false;
}
int t48(vector<int>&nums) {
	//定义 名人 所有人都认识他；他不认识所有人
	//已知函数 t48_h(i,j) 判断i是否认识j 且i==j时返回false
	//第一次遍历 找到唯一一个可能的名人
	int cond = nums[0];//名人候选人
	for (int i = 1; i < nums.size(); i++) {
		//如果i认识cond 则cond不可能是名人 更新
		if (t48_h(nums[i], cond))cond = nums[i];
	}
	//根据名人定义 验证 cond是否是名人 不是的话返回-1
	//cond是否认识其他人
	for (int i = 0; i < nums.size(); i++) {
		if (t48_h(nums[cond], nums[i]))return -1;
	}
	//其他人是否认识cond cond以后的人在第一次遍历中已经判断过
	for (int i = 0; i < cond; i++) {
		if (!t48_h(nums[i], nums[cond]))return -1;
	}
	return nums[cond];
}


//生命游戏
/*
如果活细胞周围八个位置的活细胞数少于两个，则该位置活细胞死亡；
如果活细胞周围八个位置有两个或三个活细胞，则该位置活细胞仍然存活；
如果活细胞周围八个位置有超过三个活细胞，则该位置活细胞死亡；
如果死细胞周围正好有三个活细胞，则该位置死细胞复活；
*/
void t59(vector<vector<int>>& board) {
	int m = board.size();
	int n = board[0].size();
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			//记录一个细胞周围8个位置中1的个数
			int t = 0;
			//只判读这个数的第一位
			if (j + 1 < n && (board[i][j + 1] & 1 == 1))t++;
			if (j - 1 >= 0 && (board[i][j - 1] & 1 == 1))t++;
			if (i + 1 < m && (board[i + 1][j] & 1 == 1))t++;
			if (i - 1 >= 0 && (board[i - 1][j] & 1 == 1))t++;
			if (i - 1 >= 0 && j + 1 < n && (board[i - 1][j + 1] & 1 == 1))t++;
			if (i + 1 < m&&j + 1 < n && (board[i + 1][j + 1] & 1 == 1))t++;
			if (i - 1 >= 0 && j - 1 >= 0 && (board[i - 1][j - 1] & 1 == 1))t++;
			if (i + 1 < m&&j - 1 >= 0 && (board[i + 1][j - 1] & 1 == 1))t++;
			//如果下一轮这个位置是1的话，就把这个数字的第二位改成1
			if (t == 3)board[i][j] = board[i][j] | 2;
			if (t == 2 && board[i][j] == 1)board[i][j] = board[i][j] | 2;
		}
	}
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			//通过右移一位 把这个数的第二位（下一轮轮的结果）变成第一位（当前的结果） 
			board[i][j] = board[i][j] >> 1;
		}
	}
	return;
}

//至多包含k个不同字符的最长子串 滑动窗口
//滑动窗口扩展和收缩的判断依据 map中字符的种类
int t51(string &s, int k) {
	map<char, int> m;
	int l = 0, r = 0;
	int res = 0;
	while (r < s.size()) {
		while (r < s.size() && m.size() < k) {
			m[s[r]]++;
			r++;
		}
		res = max(res, r - l);
		while (l < s.size() && m.size() == k) {
			m[s[l]]--;
			if (m[s[l]] == 0) {
				m.erase(s[l]);
				l++;
				break;
			}
			l++;
		}
	}
	return res;
}

//魔法师过河
/*
[0,4,7] 这里没有石头 变红代价4 变蓝代价7
[1,x,x] 这里红色石头 不能改变颜色
[2,x,x] 这里蓝色石头 不能改变颜色
返回一半红色一半蓝色的最小代价 如果做不到返回-1
*/
int t52(vector<vector<int>>& nums) {
	if (nums.size() % 2 == 1)return -1;
	int redstone = 0;
	int bluestone = 0;
	int redcost = 0;//所有没有颜色的石头变成红色的代价
	int bluecost = 0;//所有没有颜色的石头变成蓝色的代价
	for (int i = 0; i < nums.size(); i++) {
		if (nums[i][0] == 1)redstone++;
		if (nums[i][0] == 2)bluestone++;
		else {
			redcost += nums[i][1];
			bluecost += nums[i][2];
		}
	}
	if (redstone > nums.size() / 2 | bluestone > nums.size() / 2)return -1;
	if (redstone == bluestone && redstone == nums.size() / 2)return 0;
	if (redstone == nums.size() / 2)return bluecost;
	if (bluestone == nums.size() / 2)return redcost;
	int m = nums.size() / 2 - redstone;
	int n = nums.size() / 2 - bluestone;
	//红蓝颜色都不够一半的时候 红色石头差m个 蓝色石头差n个
	//把所有空白石头染成红色，计算代价 redcost
	//在从这些代价中减去 染成红色和染成蓝色代价相差最大的前n个石头的代价差

	//这里就认为把空白的全部染成红色 代价已经计算出 redcost,
	//也可以认为把空白的全部染成蓝色 代价已经计算出 bluecost,

	//计算每个空白石头染成红色和蓝色的代价差
	vector<int> diff;
	for (int i = 0; i < nums.size(); i++) {
		if (nums[i][0] == 0) {
			diff.push_back(nums[i][1] - nums[i][2]);
		}
	}
	//降序排列
	sort(diff.begin(), diff.end(), [](int a, int b) {return a > b; });
	for (int i = 0; i < n; i++) {
		redcost -= diff[i];
	}
	return redcost;
}

//0（1）添加 删除 获得随机元素 vector + 哈希表
/*
插入操作时，首先判断val 是否在哈希表中，如果已经存在则返false，如果不存在则插入val，操作如下：
在变长数组的末尾添加val；
在添加val 之前的变长数组长度为val 所在下标 index，将val 和下标index 存入哈希表；
返回true。

删除操作时，首先判断val 是否在哈希表中，如果不存在则返回false，如果存在则删除val，操作如下：
从哈希表中获val 的下标index；
将变长数组的最后一个元素last 移动到下标index 处，在哈希表中将last 的下标更新为index；
在变长数组中删除最后一个元素，在哈希表中删除 val；（数组最后一个元素可以不删除，对应的插入时要稍作调整，如下代码）
返回true。
*/
class RandomizedSet {
public:
	vector<int> num;
	unordered_map<int, int> m;
	int size;//下一个插入元素在变长数组中的下标，同时代表数组的有效长度
	RandomizedSet() {
		size = 0;
	}

	bool insert(int val) {
		if (m.count(val))return false;
		m[val] = size;
		if (num.size() <= size)num.push_back(val);
		else num[size] = (val);
		size++;
		return true;
	}

	bool remove(int val) {
		if (m.count(val)) {
			size--;
			/*关键点
			从哈希表中获val 的下标index；
			将变长数组的最后一个元素last 移动到下标index 处，在哈希表中将last 的下标更新为index；
			*/
			int index = m[val];
			num[index] = num[size];
			m[num[index]] = index;

			m.erase(val);
			return true;
		}
		else return false;
	}

	int getRandom() {
		return num[rand() % (size)];
	}
};
//字典序   方法数  长度k
//给定一个字符串 和 长度 返回这个字符串是总序列中的第几个（按字典序排）
//所有字符都是小写字母 比如长度为4的前几个字符串为：
//a aa aaa aaaa aaab ... aaaz ... azzz b ba baa baaa ... bzzz c
//a是这个序列中的第一个，aaab是第4个
int t53(string str, int k) {
	int res = 0;
	for (int i = 0; i < str.size(); i++) {
		res = res + (str[i] - 'a')*pow(26, k - i - 1) + 1;
	}
	return res;
}

/*
给定一个数组，当拿走某个数a的时候，其他所有的数字都加a
比如 [2,3,1]
拿走3时得3分，剩余[5,4]
拿走5时得5分，剩余9
拿走9时得9分 总计得分3+5+9 = 17
对于给定的数组，返回最大的得分
*/
int t54(vector<int> nums) {
	//排序。每次拿最大的
	sort(nums.begin(), nums.end(), [](int a, int b) {return a > b; });
	//推公式 排序后 3 2 1 拿走一个数的得分为 3 8 17 公式 res*2 + nums[i];
	int res = 0;
	for (int i = 0; i < nums.size(); i++) {
		res = res * 2 + nums[i];
	}
	return res;
}

//返回长度为k的所有子序列中 字典序最大的子序列   子序列可以不连续
string t55(string str, int k) {
	//单调栈  
	//1 在遍历到第i个的字符时 栈内的元素 加上剩余的所有元素都不大于k了 直接把栈内元素和剩余元素作为结果返回
	//2 遍历到最后 栈内的元素数目都大于等于k 则取前k个作为结果返回
	stack<char> s;
	int i;
	for (i = 0; i < str.size(); i++) {
		while (!s.empty() && str[i] >= s.top() && (s.size() + str.size() - i + 1) > k) {
			s.pop();
		}
		s.push(str[i]);
	}
	string ans;
	if (s.size() >= k) {
		while (!s.empty()) {
			ans = ans + s.top();
			s.pop();
		}
		reverse(ans.begin(), ans.end());
		return ans.substr(0, k);
	}
	else {
		while (!s.empty()) {
			ans = ans + s.top();
			s.pop();
		}
		reverse(ans.begin(), ans.end());
		for (i; i < str.size(); i++) {
			ans = ans + str[i];
		}
		return ans;
	}
	return "0";
}

/*
数组中只有 0 和 1 nums[i]的价值 取决于其左侧相邻的数字是否和它自己相同
如果 相同nums[i]的价值 等于nums[i-1]的价值+1
如果 不相同 nums[i]的价值为1
可以删除任意字符，返回整个数组的最大价值
*/
//在 nums[index...]上做选择，当前来到 Index ,前一个数字是 lastnum ,已经累积的价值为 basevalue
int t56_dfs(vector<int>& nums, int index, int lastnum, int basevalue) {
	if (index == nums.size())return 0;

	int curvalue = nums[index] == lastnum ? (basevalue + 1) : 1;

	//保留当前数字
	int p1 = t56_dfs(nums, index + 1, nums[index], curvalue);
	//不要当前数字
	int p2 = t56_dfs(nums, index + 1, lastnum, basevalue);

	return max(p1 + curvalue, p2);
}
int t56(vector<int>& nums) {
	return t56_dfs(nums, 0, 0, 0);
}

/*
数组只有0 1 2 3 四种数字，
0 1 顺序可以消除 2 3 顺序可以消除
消除后其他数字自动贴近
如果某个子序列可以全部消掉 称为全消子序列 返回全消子序列的最大长度
0231-->01-->全消
*/
//在nums[l,r]上做消除 得到的最长全消子序列
int t57_dfs(vector<int>& nums, int l, int r) {
	if (l == r) return 0;
	if (l + 1 == r) {
		if ((nums[l] == 0 && nums[r] == 1) || (nums[l] == 2 && nums[r] == 3))return 2;
		return 0;
	}
	//能消掉的子序列不考虑nums[l]的情况
	int p1 = t57_dfs(nums, l + 1, r);
	//考虑nums[i]的情况
	//如果这种情况则直接返回
	if (nums[l] == 1 || nums[l] == 3)return p1;
	//如果nums[i]可能是0或者2 则 需要找可以消掉的 1 或者 3
	int find = 0;
	if (nums[l] == 0)find = 1;
	if (nums[l] == 2)find = 3;
	int p2 = 0;
	for (int i = l + 1; i <= r; i++) {
		if (nums[i] == find) {
			p2 = max(p2, t57_dfs(nums, l + 1, i - 1) + 2 + t57_dfs(nums, i + 1, r));
		}
	}
	return max(p1, p2);
}
int t57(vector<int>& nums) {
	return t57_dfs(nums, 0, nums.size() - 1);
}

/*
一个无须数组长度n，所有数字都不一样且值都在[0..n-1]之中
返回让这个数组变有序的最小交换次数

循环下标比较
*/
int t58(vector<int>& nums) {
	int changes = 0;
	for (int i = 0; i < nums.size(); i++) {
		while (nums[i] != i) {
			swap(nums[i], nums[nums[i]]);
			changes++;
		}
	}
	return changes;
}

/*
离散化 对于一组不含有相同数字的数组 （共有n个数字） 离散化为[0,n-1] 的方法
*/
void t59(vector<int>&nums) {
	vector<int> copy = nums;
	sort(copy.begin(), copy.end());
	map<int, int> m;
	for (int i=0; i < nums.size(); i++) {
		m.insert(pair<int, int>(copy[i], i));
	}
	for (int i = 0; i < nums.size(); i++) {
		nums[i] = m[nums[i]];
	}
}

//下一个排列
void t60(vector<int>& nums) {
	//1 从后向前找第一个不是升序的数字 找到的不是升序的数nums[i-1]
	int i = 0;
	for (i = nums.size() - 1; i > 0; i--) {
		if (nums[i] > nums[i - 1])break;
	}
	if (i == 0) {
		//从后到前都是升序 则数组修改为逆序
		reverse(nums.begin(), nums.end());
		return;
	}
	//2 从最后开始找第一个大于这个数的数
	int j = 0;
	for (j = nums.size() - 1; j > i; j--) {
		if (nums[j] > nums[i - 1])break;
	}
	//3 找到的这两数交换位置
	swap(nums[i - 1], nums[j]);
	//4 第一个不是升序的数字后面的数字逆序
	reverse(nums.begin() + i, nums.end());
}

//最佳聚会地点
/*
二维网格中只有0和1   1代表一个人
有一队人（两人或以上）想要在一个地方碰面，
他们希望能够最小化他们的  总行走距离。每个人只能走直线。
官方：
两个方向的坐标是独立的，独立考虑
然后在  中位数  的点是总距离最近的
按序搜集横纵坐标，双指针，两端点相减的距离累加
*/
int t61(vector<vector<int>>& nums) {
	int m = nums.size(), n = nums[0].size();
	vector<int> x;//每个1所在的行
	vector<int> y;//每个1所在的列
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (nums[i][j]) {
				x.push_back(i);
				y.push_back(j);
			}
		}
	}
	int res = 0, l = 0, r = x.size() - 1;
	while (l < r) {
		//x[r] - x[l] 表示他想要见面的最短总路径
		res = res + x[r] - x[l];
		r--;
		l++;
	}
	l = 0, r = y.size() - 1;
	while (l < r) {
		res = res + y[r] - y[l];
		r--;
		l++;
	}
	return res;
}


/*
让数组的累加和小于等于0
可以对数组中的数执行以下操作 且每个数最多能执行一次操作
nums[i]-->0          代价x
nums[i]-->-nums[i]   代价y
返回达到目标的最小代价
*/
//来到index时 剩余的累加和
int t62_dfs(vector<int>& nums, int x, int y, int index, int sum) {
	if (sum <= 0)return 0;
	//sum还大于0 但是已经到了数组的最后 说明无效
	if (index == nums.size())return INT_MAX;
	//当前数字不变 
	int p1 = t62_dfs(nums, x, y, index + 1, sum);
	//当前数字变0
	int p2 = t62_dfs(nums, x, y, index + 1, sum - nums[index]);
	if (p2 != INT_MAX) {
		p2 = p2 + x;
	}
	//当前数字变相反数
	int p3 = t62_dfs(nums, x, y, index, sum - nums[index] * 2);
	if (p3 != INT_MAX) {
		p3 = p3 + y;
	}
	return min(p1, min(p2, p3));
}
int t62(vector<int>& nums, int x, int y) {
	int sum = 0;
	for (int c : nums) {
		sum = sum + c;
	}
	return t62_dfs(nums, x, y, 0, sum);
}

//nums中等于k种数字的子数组的个数
/*
和普通的滑动窗口解法的不同之处在于，我们需要记录两个左指针 l1 l2
第一个左指针表示包含 k 个不同整数的区间的左端点，
第二个左指针则表示包含 k-1 个不同整数的区间的左端点。
*/
int t63(vector<int>& nums, int k) {
	map<int, int> m1;
	map<int, int> m2;
	int n = nums.size();
	int l1 = 0, l2 = 0, r = 0;
	int ans = 0;
	while (r < n) {
		m1[nums[r]]++;
		m2[nums[r]]++;
		r++;
		//注意这里写大于 且内部不管是否跳出下面的while 其左端点都要 +1
		while (m2.size() > k - 1) {
			m2[nums[l2]]--;
			if (m2[nums[l2]] == 0) {
				m2.erase(nums[l2]);
				l2++;
				break;
			}
			l2++;
		}
		while (m1.size() > k) {
			m1[nums[l1]]--;
			if (m1[nums[l1]] == 0) {
				m1.erase(nums[l1]);
				l1++;
				break;
			}
			l1++;
		}
		//右端点每移动一下，都要更新一下答案
		ans = ans + l2 - l1;
	}
	return ans;
}

//魔法搭积木 要求如下
/*
上层 - 下层 <= x
一共有k个魔法积木 其重量可以视作任意值
nums数组是每个积木的重量
返回 最少能做到几堆
*/
int t64(vector<int>& nums, int k, int x) {
	//贪心 1 数组排序 再没有魔法积木的情况下 可以有几堆
	sort(nums.begin(), nums.end());
	//记录每相邻的 两堆之间的差值 所 需要的魔法积木的数量
	vector<int> needs;
	//一共有多少堆
	int buckt = 0;
	for (int i = 1; i < nums.size(); i++) {
		if (nums[i] - nums[i - 1] > x) {
			buckt++;
			needs.push_back((nums[i] - nums[i - 1] - 1) / x);
		}
	}
	//贪心2 哪些堆之间需要的魔法积木少，就先把魔法积木用在哪里
	sort(needs.begin(), needs.end());
	buckt = needs.size() + 1;
	int i = 0;
	while (buckt > 1 && k > 0) {
		k = k - needs[i];
		buckt--;
		i++;
	}
	return buckt;
}

int t64_dfs(vector<int>& nums, int index, int x, int k) {
	if (index == nums.size())return 1;
	//一定在一起
	if (nums[index + 1] - nums[index] <= x)return t64_dfs(nums, index, x, k);
	else {
		//分开
		int p1 = t64_dfs(nums, index + 1, x, k) + 1;
		//用魔法积木填充
		int p2 = INT_MAX;
		int needs = (nums[index + 1] - nums[index] - 1) / x;
		if (k >= needs)p2 = t64_dfs(nums, index + 1, x, k - needs);
		return min(p1, p2);
	}
}

//功暖器
//你找出并返回可以覆盖所有房屋的最小加热半径。
//说明：所有供暖器都遵循你的半径标准，加热的半径也一样。
int t65(vector<int>& houses, vector<int>& heaters) {
	sort(houses.begin(), houses.end());
	sort(heaters.begin(), heaters.end());
	//贪心 双指针不回退
	int i = 0, j = 0;
	int res = 0;
	for (i; i < houses.size(); i++) {
		int tem = abs(heaters[j] - houses[i]);
		//不回退
		for (j; j < heaters.size() - 1; j++) {
			//出现相等时继续向右移动，找下一个供暖器 一直到大于为止
			if (abs(heaters[j + 1] - houses[i]) <= tem) {
				tem = abs(heaters[j + 1] - houses[i]);
			}
			else {
				break;
			}
		}
		res = max(res, tem);
	}
	return res;
}

//找到所有的数字对差值的第k小的数
bool t66_h(vector<int>& nums, int mid, int k) {
	//差值小于等于mid的个数是否小于k
	//双指针 不回退
	int l = 0, r = 1;
	int count = 0;
	for (l; l < nums.size(); l++) {
		r = max(r, l);
		while (r < nums.size() && abs(nums[r] - nums[l]) <= mid) {
			count++;
			r++;
		}
	}
	return count < k;
}
/*
关于二分  伪代码
l = -1,r = N;
while((l+1)!=r){
	mid = l + (r-l)/2;
	if isblue(m) l = mid;
	else r = mid;
}
return l or r;

例子 1 2 3 5 5 5 8 9
					   条件   return
第一个 >= 5   的元素     <5       r         lower_bound()
最后一个 < 5  的元素     <5       l
第一个 > 5    的元素     <=5      r         upper_bound()
最后一个 <= 5 的元素     <=5      l
*/
int t66(vector<int>& nums, int k) {
	//所有的数字对个数 即差值个数
	int m = (nums.size()*nums.size() - 1) / 2;
	//无效
	if (m < k || m < 1 || k < 1)return -1;
	sort(nums.begin(), nums.end());
	//二分查找
	int l = -1;
	int r = nums.back() - nums[0];
	while (l != r - 1) {
		int mid = l + (r - l) / 2;
		if (t66_h(nums, mid, k))l = mid;
		else r = mid;
	}
	return l;
}

int t67numberOfArithmeticSlices(vector<int>& nums) {
	//事先定义好数组大小 程序运行时间更短
	vector<unordered_map<int, int>>m(nums.size());
	unordered_map<int, int> mpre;
	int res = 0;
	for (int i = 0; i < nums.size(); i++) {
		for (int j = i - 1; j >= 0; j--) {
			long mp = (long)nums[i] - (long)nums[j];
			if (mp > INT_MAX || mp < INT_MIN)continue;
			mp = (int)mp;
			//习惯使用find 和智能指针 auto
			auto it = m[j].find(mp);
			//首先计算以某一个数字结尾的至少为两个数字的子序列
			//以某一个数字结尾的至少为三个数字的子序列个数比两个数字的子序列个数少一个
			//nums[i]和nums[j]的差值 在nums[j]的哈希表中有几个
			int cnt = it == m[j].end() ? 0 : it->second;
			res = res + cnt;
			//nums[i]的哈希表中这个差值的个数为 原有的个数 + nums[j]的哈希表中的个数 + 1
			//这样是合法的
			m[i][mp] = m[i][mp] + 1 + cnt;
		}
	}
	return res;
}

//最长公共子序列问题  不要求连续
int t_68maxUncrossedLines(vector<int>& nums1, vector<int>& nums2) {
	vector<vector<int>>dp(nums1.size() + 1, vector<int>(nums2.size() + 1));
	for (int i = 0; i < nums1.size(); i++)dp[i][0] = 0;
	for (int i = 0; i < nums2.size(); i++)dp[0][i] = 0;
	for (int i = 1; i <= nums1.size(); i++) {
		for (int j = 1; j <= nums2.size(); j++) {
			//dp[i][j] nums1的前i个数字和nums2的前j个数字
			//nums1的第i个数字 nums1[i-1]
			//nums2的第j个数字 nums2[j-1]
			if (nums1[i - 1] == nums2[j - 1])dp[i][j] = dp[i - 1][j - 1] + 1;
			else dp[i][j] = max(dp[i][j - 1], dp[i - 1][j]);
		}
	}
	return dp[nums1.size()][nums2.size()];
}

//爱吃香蕉的珂珂
bool t69_check(vector<int>& piles, int h, int v) {
	int res = 0;
	for (int i : piles) {
		res = res + i / v + (i%v == 0 ? 0 : 1);
	}
	return res <= h;
}
int t69minEatingSpeed(vector<int>& piles, int h) {
	int res = 0;
	for (int c : piles) {
		res = max(res, c);
	}
	if (piles.size() == h) {
		return res;
	}
	else {
		//二分
		int l = 1, r = res + 1;
		while (l < r) {
			int mid = l + (r - l) / 2;
			//速度为mid时 h小时能不能吃完piles堆香蕉
			if (t69_check(piles, h, mid))r = mid;
			else l = mid + 1;
		}
		return l;
	}
	return 0;
}

struct t70mycmp {
	bool operator()(vector<int>& a, vector<int>& b) {
		return a[1] > b[1];
	}
};
/*
nums[i]=j  第 i 天 j 号湖泊下雨  j==0 所有湖泊不下雨
//返回一个数组 每天抽的几号湖泊的水
//下雨天 不抽水 填-1  没有水可抽的不下雨天 填1
大思路 如果某一天不下雨 则优先抽取右边离得最近的已经下过雨的湖泊
关键点 维护一个优先级队列 下一次要抽水的湖泊号
*/
vector<int> t70avoidFlood(vector<int>& rains) {
	//遍历一遍 记录每个湖泊的下雨时间
	unordered_map<int, list<int>> time;
	for (int i = 0; i < rains.size(); i++) {
		time[rains[i]].push_back(i);
	}
	//一个表记录当前下过雨的湖泊
	unordered_set<int> water;
	//一个优先队列 保存要抽干水的湖泊顺序 key号湖下次的下雨时间value 
	//按照value升序排列
	priority_queue<vector<int>, vector<vector<int>>, t70mycmp> p;
	vector<int> ans;
	for (int i = 0; i < rains.size(); i++) {
		if (rains[i] == 0) {
			//可以去抽水
			//有水可抽
			if (p.size()) {
				//要抽几号湖
				int lake = p.top()[0];
				//从优先队列中删除
				p.pop();
				ans.push_back(lake);
			}
			//没有水可抽
			else {
				ans.push_back(1);
			}
		}
		else {
			//不能抽水 记录下雨的湖泊
			//如果 这个湖泊还有水 却又下雨了 失败返回空
			int lake = rains[i];
			if (water.count(lake))return {};
			//如果这个湖泊没水 下雨了 添加记录
			water.insert(lake);
			//把这个湖的下次的下雨时间添加进优先队列
			//cout<<lake<<" "<<time[lake].front()<<endl;
			p.push({ lake,time[lake].front() });
			time[lake].pop_front();
			if (!time[lake].size()) {
				//这个湖后面不在会下雨了 删除
				time.erase(lake);
			}
			ans.push_back(-1);
		}
	}
	return ans;
}

/*
436 寻找右区间
给你一个区间数组 intervals ，其中 intervals[i] = [starti, endi] ，且每个 starti 都 不同 。
区间 i 的 右侧区间 可以记作区间 j ，并满足 startj >= endi ，且 startj 最小化 。
返回一个由每个区间 i 的 右侧区间 在 intervals 中对应下标组成的数组。如果某个区间 i 不存在对应的 右侧区间 ，则下标 i 处的值设为 -1

*/
vector<int> t71findRightInterval(vector<vector<int>>& intervals) {
	map<int, int>m;//起始点 ：在intervals中的index
	for (int i = 0; i < intervals.size(); i++) {
		m[intervals[i][0]] = i;
	}
	vector<int> ans;
	for (int i = 0; i < intervals.size(); i++) {
		//找第一个>= Intervals[i][1]的st
		//lower_bound在map中的使用
		auto it = m.lower_bound(intervals[i][1]);
		if (it != m.end()) {
			ans.push_back(it->second);
		}
		else {
			ans.push_back(-1);
		}
	}
	return ans;
}

/*
大顶堆 用数组代替完全二叉树模拟
对于一个结点下标 index 父节点 (index-1)/2
				   左孩子 2*index +1
				   右孩子 2*index +2
*/
class myheap {
	//已经收集到的元素个数，下一个元素在arr中的位置
	int size;
	int limit;
	vector<int> arr;
	myheap(int m) {
		limit = m;
		size = 0;
		arr = vector<int>(limit, 0);
	}
	bool isempty() {
		return size == 0;
	}
	bool isfull() {
		return size == limit;
	}
	void heapInsert(int index) {
		//在完全二叉树中 arr[index] 的父节点 arr[(index-1)/2]
		//不断的和他的父节点比较大小 直到他不大于他父节点未知 或者 他是 0
		while (arr[index] > arr[(index - 1) / 2]) {
			swap(arr[index], arr[(index - 1) / 2]);
			index = (index - 1) / 2;
		}
	}
	bool push(int v) {
		if (size == limit)return false;
		else {
			arr[size] = v;
			//把这个数字尽可能的 往树的上面插
			heapInsert(size);
			size++;
			return true;
		}
	}
	void heapify(int index, int size) {
		//不断向下找，index与他的左右孩子中比他大的中最大的孩子交换
		//直到 左右孩子都不大于他为止 或者没有孩子
		int left = index * 2 + 1;
		while (left < size) {
			int right = left + 1;
			//向下找他和他左右儿子中最大数在数组中的下标
			int largest = INT_MIN;
			if (right < size) {
				largest = arr[left] > arr[index] ? left : index;
				largest = arr[largest] > arr[right] ? largest : right;
			}
			else largest = arr[left], arr[index]?left:index;
			//如果结果是他自己 则结束
			if (largest == index)return;
			//否则 交换位置，更新继续往下找
			swap(arr[index], arr[largest]);
			index = largest;
			left = index * 2 + 1;
		}
	}
	int pop() {
		int res = arr[0];
		size--;
		//删掉最大值 实际上是用数组的最后一个数字把第一个数组覆盖掉
		swap(arr[0], arr[size]);
		//此时 size 数组最后一个元素的位置
		//把0位置的数一直向树的下面找，合适的位置
		heapify(0, size);
		return res;
	}
};

//两个人比赛，分别为1号和2号 
//一堆糖果，谁先没得吃谁输，每个人每次只能吃 1/4/16/64。。。个糖果，两者都是绝对聪明的
//1号先手 返回赢家 1/2
int t72winner(int N) {
	if (N == 0 || N == 2 || N == 5)return 2;
	if (N == 1 || N == 3 || N == 4)return 1;
	int base = 1;
	while (base <= N) {
		//1号吃了后，在调用函数的时候1号在里面为后手
		if (2 == t72winner(N - base))return 1;
		base = base * 4;
	}
	return 2;
}


// Z 字打印矩阵
void t73_h(vector<vector<int>>& nums, bool fromup,int ar,int ac,int br,int bc) {
	if (fromup) {
		while (ac != bc) {
			cout << nums[ar][ac];
			ar++;
			ac--;
		}
	}
	else {
		while (ac != bc) {
			cout << nums[br][bc];
			br--;
			bc++;
		}
	}
}
void t73(vector<vector<int>>& nums) {
	int M = nums.size();
	int N = nums[0].size();
	//A B两点初始坐标
	int Ar = 0,Ac = 0;
	int Br = 0,Bc = 0;
	bool fromup = false;
	while (Ar < M) {
		//A点一直向右移动，直到最后一列 在向下移动
		Ar = Ac == N - 1 ? Ar + 1 : Ar;
		Ac = Ac == N - 1 ? Ac : Ac + 1;
		//B点一直向下移动 直到最后一行 在向右移动
		Bc = Br == M - 1 ? Bc + 1 : Bc;
		Br = Br == M - 1 ? Br : Br + 1;
		t73_h(nums, fromup,Ar,Ac,Br,Bc);
		fromup = !fromup;
	}
}

//top面试 T1
vector<int> twoSum(vector<int>& nums, int target) {
	map<int, int> m;
	//只记录一个数字第一次出现的位置
	for (int i = 0; i < nums.size(); i++) {
		if (m.count(nums[i]))continue;
		else {
			m[nums[i]] = i;
		}
	}
	//找target-x的位置 且 target-x 的位置不能和 x 的位置相同
	for (int i = 0; i < nums.size(); i++) {
		if (m.count(target - nums[i]) && m[target - nums[i]] != i)return { m[target - nums[i]],i };

	}
	return {};
}

//top面试 T2
Lnode* addTwoNumbers(Lnode* l1, Lnode* l2) {
	Lnode * tmp = nullptr;
	int cnt = 0;
	Lnode * root = new Lnode((cnt + l1->val + l2->val) % 10);
	cnt = (cnt + l1->val + l2->val) / 10;
	tmp = root;
	l2 = l2->next;
	l1 = l1->next;
	while (l2&&l1) {
		tmp->next = new Lnode;
		tmp->next->val = (cnt + l1->val + l2->val) % 10;
		cnt = (cnt + l1->val + l2->val) / 10;
		tmp = tmp->next;
		l2 = l2->next;
		l1 = l1->next;
	}
	while (l1) {
		tmp->next = new Lnode;
		tmp->next->val = (cnt + l1->val) % 10;
		cnt = (cnt + l1->val) / 10;
		tmp = tmp->next;
		l1 = l1->next;
	}
	while (l2) {
		tmp->next = new Lnode;
		tmp->next->val = (cnt + l2->val) % 10;
		cnt = (cnt + l2->val) / 10;
		tmp = tmp->next;
		l2 = l2->next;
	}
	if (cnt > 0) {
		tmp->next = new Lnode;
		tmp->next->val = cnt;
	}
	return root;
}

//top面试 T3
//最长无重复字符的子串长度
int t6(string s) {
	//每个字符上一次出现的位置，初始为-1
	vector<int> map(256, -1);
	//子串，要求连续，以第 i 个字符结尾的最长无重复子串长度
	vector<int> dp(s.size(), 0);
	int ans = 1;
	dp[0] = 1;
	map[s[0]] = 0;
	for (int i = 1; i < s.size(); i++) {
		//1 以当前字符为最后位置，往前推的的最远距离
		//2 以上一个字符为最后的最长无重复子串长度
		dp[i] = min(i - map[s[i] - 'a'], dp[i - 1] + 1);
		ans = max(ans, dp[i]);
		map[s[i] - 'a'] = i;
	}
	return ans;
}
int lengthOfLongestSubstring(string s) {
	unordered_set<char> m;
	int l = 0, r = 0;
	int res = 0;
	while ( r < s.size()) {
		//没有重复的
		while (r < s.size() && !m.count(s[r])) {
			m.insert(s[r]);
			r++;
		}
		res = max(res, r - l);
		//出现重复的
		while (l < s.size() && m.count(s[r])) {
			m.erase(s[l]);
			l++;
		}
	}
	return res;
}

//top面试 T4
double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
	//找到较短的数组 使其为nums1
	if (nums1.size() <= nums2.size());
	else {
		vector<int> t = nums2;
		nums2 = nums1;
		nums1 = t;
	}
	//对较短的数组 nums1 二分 根据二分的位置 确定nums2二分的位置
	//判断是否满足交叉小于等于 不满足继续二分
	int n = nums1.size(), m = nums2.size(), len = m + n;
	int l = 0, r = n;
	while (l <= r) {
		int mid = l + (r - l) / 2;
		//根据第一个二分的位置 确定第二个的二分位置
		int mid2 = (len + 1) / 2 - mid;
		//把二分的位置的数字当作右边数字，二分的前一个位置当作左边的数字
		//如果左边没字符了，就定义成最小值，让所有数都大于它，否则就是nums1二分的位置左边一个
		//如果右边没字符了，就定义成最大值，让所有数都小于它，否则就是nums1二分的位置
		double L1 = mid == 0 ? INT_MIN : nums1[mid - 1];
		double R1 = mid == n ? INT_MAX : nums1[mid];
		double L2 = mid2 == 0 ? INT_MIN : nums2[mid2 - 1];
		double R2 = mid2 == m ? INT_MAX : nums2[mid2];

		//需要满足 L1<=R2&&L2<=R1

		//不满足L1<=R2 nums1在左侧二分
		if (L1 > R2) {
			r = mid - 1;
		}
		//不满足L2<=R1 nums1在右侧二分
		else if (L2 > R1) {
			l = mid + 1;
		}
		//满足时 返回
		//长度为偶数返回作  左侧较大者  和  右边较小者  和的一半
		//长度为奇数返回作左侧较大者
		else {
			if (len % 2 == 0)return (max(L1, L2) + min(R1, R2)) / 2;
			else return max(L1, L2);
		}
	}
	return 0.0;
}

//top面试 T5
string tar(string s) {
	string t;
	for (int i = 0; i < s.size(); i++) {
		if (i == 0)t.push_back('*');
		t.push_back(s[i]);
		t.push_back('*');
	}
	return t;
}
string longestPalindrome(string s) {
	//jio-->*j*i*o*
	s = tar(s);
	cout << s << endl;
	//p[i] 以i为中心的构成的回文串的半径+1
	vector<int> p(s.size());
	int c = -1, r = -1;
	int res = 0;
	for (int i = 0; i < s.size(); i++) {
		// 2*c-i 是i关于c对称的位置
		p[i] = r > i ? min(r - i, p[2 * c - i]) : 1;
		while (i - p[i] >= 0 && i + p[i] < s.size()) {
			if (s[i - p[i]] == s[i + p[i]])p[i]++;
			else break;
		}
		if (p[i] + i > r) {
			r = i + p[i];
			c = i;
		}
		if (p[i] > p[res]) {
			res = i;
		}
		cout << p[i] << endl;
	}
	string ans;
	string tmp = s.substr(res - p[res] + 1, 2 * (p[res] - 1));
	for (char c : tmp) {
		if (c != '*')ans.push_back(c);
	}
	return ans;
}
//top面试 T7
int reverse(int x) {
	int res = 0;
	while (x) {
		int t = x % 10;
		x = x / 10;
		//提前判断溢出 在不会有溢出的情况的下，进行下一步
		if (res > INT_MAX / 10 || res < INT_MIN / 10)return 0;
		res = res * 10 + t;
	}
	return res;
}

//top面试 T10
bool isMatch(string s, string p) {
	//p不为空 s为空时 还需要继续进行匹配
	if ((s.size() != 0 && p.size() == 0))return false;
	if (s.size() == 0 && p.size() == 0)return true;
	//判断第一个字符是否相同
	bool firstMatch = (s.size() > 0 && (s[0] == p[0] || p[0] == '.'));
	//如果p的第二个字符为*
	if (p.size() >= 2 && p[1] == '*') {
		//s删除第一个字符 然后递归 *匹配非0次（要求 第一个字符相同）
		//p删除前两个字符 然后递归 *匹配0次
		return (firstMatch&&isMatch(s.substr(1), p)) || (isMatch(s, p.substr(2)));
	}
	//更一般的情况 一个一个比较
	else {
		return (firstMatch&&isMatch(s.substr(1), p.substr(1)));
	}
}
//记忆化搜索优化
bool isMatch1(string & s, string & p, int i, int j, vector<vector<int>>& dp) {
	if ((i < s.size() && j == p.size()))return false;
	if (i == s.size() && j == p.size())return true;
	//dp[i][j] 当递归来到 s 的第 i 个字符， p 的第 j 个字符时
	//dp[i][j] 0表示不知道，1表示匹配，2表示不匹配
	if (dp[i][j] != 0) {
		return dp[i][j] == 1 ? true : false;
	}
	bool firstMatch = (i < s.size() && (s[i] == p[j] || p[j] == '.'));
	if (p.size() - j >= 2 && p[j + 1] == '*') {
		bool p1 = (firstMatch&&isMatch1(s, p, i + 1, j, dp)) || (isMatch1(s, p, i, j + 2, dp));
		dp[i][j] = p1 == true ? 1 : 2;
		return p1;
	}
	else {
		bool p2 = (firstMatch&&isMatch1(s, p, i + 1, j + 1, dp));
		dp[i][j] = p2 == true ? 1 : 2;
		return p2;
	}
}

//top面试 T11
int maxArea(vector<int>& height) {
	int l = 0, r = height.size() - 1;
	int res = min(height[l], height[r])*(r - l);
	while (l < r) {
		if (height[l] < height[r])l++;
		else r--;
		res = max(res, min(height[l], height[r])*(r - l));
	}
	return res;
}

//top面试 T13
/*
若存在小的数字在大的数字的左边的情况，根据规则需要减去小的数字。
对于这种情况，我们也可以将每个字符视作一个单独的值，若一个数字右侧的数字比它大，则将该数字的符号取反。
例如 XIV 可视作 {X}-{I}+{V}=10-1+5=14。
*/
int romanToInt(string s) {
	unordered_map<char, int> m = { {'I',1},{'V',5},{'X',10},{'L',50},{'C',100},{'D',500},{'M',1000} };
	int res = 0;
	for (int i = 0; i < s.size(); i++) {
		if (i + 1 < s.size() && m[s[i]] < m[s[i + 1]]) {
			res = res - m[s[i]];
		}
		else {
			res = res + m[s[i]];
		}
	}
	return res;
}

//top面试 T14
string longestCommonPrefix(vector<string>& strs) {
	string s;
	for (int j = 0; j < strs[0].size(); j++) {
		for (int i = 0; i < strs.size(); i++) {
			if (j < strs[i].size() && strs[i][j] != strs[0][j])return s;
			if (j >= strs[i].size())return s;
		}
		s.push_back(strs[0][j]);
	}
	return s;
}

//top面试 T15
vector<vector<int>> threeSum(vector<int>& nums) {
	vector<vector<int>> ans;
	set<vector<int>> m;
	sort(nums.begin(), nums.end());
	for (int i = 0; i < nums.size(); i++) {
		int target = nums[i] * (-1);
		int l = 0, r = nums.size() - 1;
		while (l < r) {
			if (l == i)l++;
			else if (r == i)r--;
			else if (nums[l] + nums[r] < target)l++;
			else if (nums[l] + nums[r] > target)r--;
			else {
				vector<int> t = { nums[i],nums[l],nums[r] };
				sort(t.begin(), t.end());
				if (!m.count(t)) {
					ans.push_back(t);
					m.insert(t);
				}
				l++;
				r--;
			}
		}
		while (i + 1 < nums.size() && nums[i + 1] == nums[i])i++;
	}
	return ans;
}
//小优化
vector<vector<int>> threeSum2(vector<int>& nums) {
	vector<vector<int>> ans;
	//先排序
	sort(nums.begin(), nums.end());
	for (int i = 0; i < nums.size(); i++) {
		int target = nums[i] * (-1);
		//第二个数字从第一个数字后面开始枚举
		int l = i + 1, r = nums.size() - 1;
		while (l < r) {
			//需要枚举不同的第二个数字
			if (l > i + 1 && nums[l] == nums[l - 1])l++;
			else if (nums[l] + nums[r] < target) {
				l++;
			}
			else if (nums[l] + nums[r] > target) {
				r--;
			}
			else {
				{
					ans.push_back({ nums[i],nums[l],nums[r] });
					l++;
					r--;
				}
			}
		}
		//需要枚举不同的第一个数字
		while (i + 1 < nums.size() && nums[i + 1] == nums[i])i++;
	}
	return ans;
}

//top面试 T17
void f(vector<string>&res, string& digits, unordered_map<char, vector<char>>& m, int index, string str) {
	if (str.size() >= digits.size()) {
		res.push_back(str);
		return;
	}
	for (char c : m[digits[index]]) {
		str.push_back(c);
		index++;
		f(res, digits, m, index, str);
		index--;
		str.pop_back();
	}
	return;
}
vector<string> letterCombinations(string digits) {
	if (digits.size() < 1)return {};
	vector<string> res;
	unordered_map<char, vector<char>> m = { {'2',{'a','b','c'}},{'3',{'d','e','f'}},{'4',{'g','h','i'}},{'5',{'j','k','l'}},{'6',{'m','n','o'}},{'7',{'p','q','r','s'}},{'8',{'t','u','v'}},{'9',{'w','x','y','z'}} };
	f(res, digits, m, 0, "");
	return res;
}


//top面试 T19
Lnode* removeNthFromEnd(Lnode* head, int n) {
	Lnode * p1 = head;
	Lnode * p2 = head;
	//双指针 p1先走n步 
	while (n) {
		if (p1->next)p1 = p1->next;
		//此时说明要删除的实际上是第一个元素
		else {
			p2 = head->next;
			head = nullptr;
			return p2;
		}
		n--;
	}
	//p1 p2一块走 ，一直到p1走完
	while (p1->next) {
		p1 = p1->next;
		p2 = p2->next;
	}
	//p2的下一个元素是要被删除的元素
	p2->next = p2->next->next;
	return head;
}

//top面试 T21
Lnode* mergeTwoLists(Lnode* list1, Lnode* list2) {
	if (!list1 && !list2)return nullptr;
	if (!list1)return list2;
	if (!list2)return list1;
	//把两个链表中头节点数字小的链表当作list1，把list2中的元素向list1中插，
	//头节点已经确定，从list2中的第一个元素和list1中的第二个元素开始比较插入
	Lnode* head = list1->val >= list2->val ? list2 : list1;
	list2 = head == list2 ? list1 : list2;
	list1 = head->next;
	//记录插入位置的前一个结点
	Lnode * node = head;
	while (list2&&list1) {
		if (list1->val > list2->val) {
			//把一个元素插入
			Lnode * tmp = list2->next;
			list2->next = node->next;
			node->next = list2;
			//更新list1 list2 node
			list2 = tmp;
			list1 = node->next;
			node = list1;
		}
		else {
			//更新node list1
			node = list1;
			list1 = list1->next;
		}
	}
	if (list2) {
		node->next = list2;
	}
	return head;
}

//top面试 T22
void generateParenthesis_dfs(int N, vector<string>& strs, int index, string str) {
	if (index >= N) {
		//判断括号是否合法
		stack<char> f;
		for (char t : str) {
			if (t == '(')f.push(')');
			else if (t == ')') {
				if (f.empty() || f.top() != ')')return;
				else if (!f.empty() && f.top() == ')') {
					f.pop();
				}
			}
		}
		if (f.empty())strs.push_back(str);
		return;
	}
	for (char c : {'(', ')'}) {
		index++;
		str.push_back(c);
		generateParenthesis_dfs(N, strs, index, str);
		str.pop_back();
		index--;
	}
}
vector<string> generateParenthesis(int n) {
	vector<string> res;
	generateParenthesis_dfs(2 * n, res, 0, "");
	return res;
}

/*
我们可以只在序列仍然保持有效时才添加 ‘(’ ,‘)’，而不是每次添加。
我们可以通过跟踪到目前为止放置的左括号和右括号的数目来做到这一点，
如果左括号数量不大于 n，我们可以放一个左括号。如果右括号数量小于左括号的数量，我们可以放一个右括号。
*/
void generateParenthesis_dfs2(int N, vector<string>& strs, int index, string str,int l,int r) {
	if (index >= N) {
		strs.push_back(str);
		return;
	}
	if (l <= N / 2) {
		index++;
		str.push_back('(');
		generateParenthesis_dfs2(N, strs, index, str,l+1,r);
		str.pop_back();
		index--;
	}
	if (r < l) {
		index++;
		str.push_back(')');
		generateParenthesis_dfs2(N, strs, index, str, l, r+1);
		str.pop_back();
		index--;
	}
}

//top面试 T22
//直接每次当作合并两个链表也可以，但是每次合并的两个链表的长度在一直增加
struct cmp {
	//根据结点的数字大小升序排列
	bool operator ()(Lnode* a, Lnode* b) {
		return a->val > b->val;
	}

};
Lnode* mergeKLists(vector<Lnode*>& lists) {
	if (lists.size() < 1)return nullptr;
	if (lists.size() == 1)return lists[0];
	priority_queue<Lnode*, vector<Lnode*>, cmp> p;
	//记录每个链表的头节点
	for (auto node : lists) {
		if (node)p.push(node);
	}
	Lnode* head = new Lnode;
	Lnode* root = head;
	Lnode * tmp = nullptr;
	while (!p.empty()) {
		tmp = p.top();
		root->next = tmp;
		root = root->next;
		p.pop();
		if (tmp->next)p.push(tmp->next);
	}
	return head->next;
}

//top面试 T26
int removeDuplicates(vector<int>& nums) {
	int end = 0;//没有重复数字数组的最后一个位置
	int cur = 1;//当前到达的位置
	int rep = -1;//数组中发现的第一个重复数字的位置，-1时没有重复数字
	for (int cur = 1; cur < nums.size(); cur++) {
		//找到的第一个重复数字，记录位置
		if (rep == -1 && nums[end] == nums[cur]) {
			rep = end + 1;
		}
		//把重复位置的数字换掉，更新重复数字的位置为无效，更新有效的数组长度
		else if (rep != -1 && nums[end] != nums[cur]) {
			nums[rep] = nums[cur];
			end = rep;
			rep = -1;
		}
		//发现和重复数字相同的数字，不做处理
		else if (rep != -1 && nums[end] == nums[cur]) {
			continue;
		}
		//没有重复数字 且当前数字不重复，有效数组长度更新
		else {
			//如果当前数字位置和有效数组长度不统一，当前数字接在有效数组后面
			if (cur - end > 1)nums[end + 1] = nums[cur];
			end++;
		}
	}
	return end + 1;
}

////top面试 T28 KMP
int strStr(string haystack, string needle) {
	int n = haystack.size(), m = needle.size();
	if (m == 0) return 0;
	//设置哨兵
	haystack.insert(haystack.begin(), ' ');
	needle.insert(needle.begin(), ' ');
	vector<int> next(m + 1);
	//预处理next数组 构造 i 从2开始
	for (int i = 2, j = 0; i <= m; i++) {
		//j回退
		while (j != 0 && needle[i] != needle[j + 1])j = next[j];
		//j先++ 在for里面i++
		if (needle[i] == needle[j + 1])j++;
		//确定next[i]的值 数组中每个位置的值就是该下标应该跳转的目标位置
		next[i] = j;
	}
	
	//匹配过程 匹配 i 从1开始
	for (int i = 1, j = 0; i <= n; i++) {
		//j回退
		while (j != 0 && haystack[i] != needle[j + 1])j = next[j];
		//j先++ 在for里面i++
		if (haystack[i] == needle[j + 1])j++;
		//匹配到了，返回位置
		if (j == m)return i - m;
	}
	return -1;
}

/*
那么对于a*53 = a*110101（二进制）= a*（100000+10000+100+1）=a*（100000*1+10000*1+1000*0+100*1+10*0+1*1）。
那么设立一个ans=0用于保存答案，每一位让a*=2，在根据b的对应为1看是不是加上此时的a，即可完成快速运算
*/
int mul(int a, int b) {
	int ans=0;
	while (b) {
		if (b & 1 == 1)ans = ans + a;
		b = b >> 1;
		a = a + a;
	}
	return ans;
}
//top面试 T29 
bool check(int a, int m, int b) {
	//判断 a*m 是否 大于等于b 
	//注意 a b 都是负数 所以 res 从0开始减小， a 也是一直减小的 
	//所以判断 a*m 是否 大于等于b 实际是判断进行m循环后res是否仍然大于等于b
	int res = 0;
	while (m) {
		if (m & 1==1) {
			//要保证 res+a>=b
			if (res < b - a)return false;
			res = res + a;
		}
		if (m > 1) {
			// m>1 一定有下一次的加法时，要保证 a+a>=b 
			//如果 a+a<b 则到下一次用一个小于等于0的数（res）+ （a+a）一定是小于b的
			if (a < b - a)return false;
			a = a + a;
		}
		m = m >> 1;
	}
	return res >= b;
}
int divide(int dividend, int divisor) {
	if (dividend == INT_MIN) {
		if (divisor == 1)return INT_MIN;
		//此时发生越界 返回最大数
		if (divisor == -1)return INT_MAX;
	}
	if (divisor == INT_MIN) {
		if (dividend == INT_MIN)return 1;
		else return 0;
	}
	//结果是否为负数
	bool sign = (dividend > 0 && divisor < 0) || (dividend < 0 && divisor>0);
	//把数字放在负数范围处理
	if (dividend > 0)dividend = -dividend;
	if (divisor > 0)divisor = -divisor;
	int l = 1, r = INT_MAX;
	int ans = 0;
	while (l <= r) {
		int mid = l + ((r - l) >> 1);
		//判断 divisor * mid >= dividend divisor和dividend是负数
		//如果 divisor * mid >= dividend 说明 mid太小 向右括
		//否则 mid太大 向左扩
		if (check(divisor, mid, dividend)) {
			ans = mid;
			//提前判断是否会发生越界
			if (mid == INT_MAX)break;
			l = mid + 1;
		}
		else r = mid - 1;
	}
	//根据结果的符号 返回对应的ans
	return sign ? -ans : ans;
}

//top面试 T33
/*
数组实际是这样的，ab段的第一个元素会大于cd段的每一个元素
（ab段）
	7
  6
5
			   4
			3
	     2
	  1
	  （cd段）
*/
int search(vector<int>& nums, int target) {
	int len = nums.size() - 1;
	int l = 0, r = len;
	while (l <= r) {
		int mid = l + (r - l) / 2;
		if (target == nums[mid])return mid;
		//nums[mid] >= nums[0] 说明nums[mid]必定在旋转点的左边区间
		if (nums[0] <= nums[mid]) {
			//nums[0] <= target <= nums[mid] 说明target必定在[l,mid-1]
			if (nums[0] <= target && target <= nums[mid])r = mid - 1;
			else l = mid + 1;
		}
		//nums[mid] < nums[0] 说明nums[mid]必定在旋转点的右边区间
		else {
			//若nums[mid]<=target<=nums[right] 说明target必定在[mid+1,r]
			if (nums[mid] <= target && target <= nums[r])l = mid + 1;
			else r = mid - 1;
		}
	}
	return -1;
}

//top面试 T34
/*
关于二分  伪代码
l = -1,r = N;
while((l+1)!=r){
	mid = l + (r-l)/2;
	if isblue(m) l = mid;
	else r = mid;
}
return l or r;

例子 1 2 3 5 5 5 8 9
					   条件   return
第一个 >= 5   的元素     <5       r         lower_bound()
第一个 > 5    的元素     <=5      r         upper_bound()
最后一个 < 5  的元素     <5       l
最后一个 <= 5 的元素     <=5      l
*/
vector<int> searchRange(vector<int>& nums, int target) {
	vector<int> ans{ -1,-1 };
	int l = -1, r = nums.size();
	while (l + 1 != r) {
		int mid = l + (r - l) / 2;
		if (nums[mid] <= target)l = mid;
		else r = mid;
	}
	if (l >= 0 && l < nums.size() && nums[l] == target)ans[1] = l;
	l = -1, r = nums.size();
	while (l + 1 != r) {
		int mid = l + (r - l) / 2;
		if (nums[mid] < target)l = mid;
		else r = mid;
	}
	if (r >= 0 && r < nums.size() && nums[r] == target)ans[0] = r;
	return ans;
}

//top面试 T36
bool isValidSudoku(vector<vector<char>>& board) {
	//第i行的j有没有出现过 i:[0 8] j:[1 9]
	vector<vector<bool>> row(9, vector<bool>(10));
	//第i列的j有没有出现过
	vector<vector<bool>> col(9, vector<bool>(10));
	//第i个方块的j有没有出现过
	vector<vector<bool>> buck(9, vector<bool>(10));
	for (int i = 0; i < 9; i++) {
		for (int j = 0; j < 9; j++) {
			if (board[i][j] == '.')continue;
			int tmp = 3 * (j / 3) + i / 3;
			int num = board[i][j] - '0';
			if (row[i][num] || col[j][num] || buck[tmp][num])return false;
			else {
				row[i][num] = true;
				col[j][num] = true;
				buck[tmp][num] = true;
			}
		}
	}
	return true;
}

//top面试 T38
string countAndSay_dfs(int n, string str) {
	if (n == 1) {
		return str;
	}
	int num = 1, i = 1;
	string tmp = "";
	for (i = 1; i < str.size(); i++) {
		if (str[i] == str[i - 1])num++;
		else {
			tmp.push_back(num + '0');
			tmp.push_back(str[i - 1]);
			num = 1;
		}
	}
	//把最后一个记录添加进去
	tmp.push_back(num + '0');
	tmp.push_back(str[i - 1]);
	return countAndSay_dfs(n - 1, tmp);
}
string countAndSay(int n) {
	string s = "1";
	//从第一项开始递归
	return countAndSay_dfs(n, s);
}

//top面试 T41
/*
对于一个长度为 N 的数组，其中没有出现的最小正整数只能在 [1, N+1]中
如果数组中包含 x∈[1,N]，那么恢复后，数组的第 x−1 个元素为 x
在恢复后，数组应当有 [1, 2, ..., N] 的形式，但其中有若干个位置上的数是错误的，
每一个错误的位置就代表了一个缺失的正数。第一个数字错误的位置就是结果
以题目中的示例二 [3, 4, -1, 1] 为例，恢复后的数组应当为 [1, -1, 3, 4]，我们就可以知道缺失的数为 2。
*/
int firstMissingPositive(vector<int>& nums) {
	int n = nums.size();
	for (int i = 0; i < n; i++) {
		while (nums[i] > 0 && nums[i] <= n && (nums[i] != nums[nums[i] - 1])) {
			swap(nums[i], nums[nums[i] - 1]);
		}
	}
	for (int i = 0; i < n; i++) {
		if (nums[i] != i + 1)return i + 1;
	}
	return n + 1;
}

//top面试 T42
int trap(vector<int>& height) {
	int ans = 0;
	int lm = 0, rm = 0;
	int l = 0, r = height.size() - 1;
	while (l < r) {
		//谁小移动谁
		if (height[l] < height[r]) {
			//当前值小于左侧已经遍历过的最大值是 更新答案
			if (height[l] < lm)ans = ans + lm - height[l];
			//否则 更新左侧最大值
			else lm = height[l];
			l++;
		}
		else {
			if (height[r] < rm)ans = ans + rm - height[r];
			else rm = height[r];
			r--;
		}
	}
	return ans;
}

//top面试 T44
bool isMatch_dfs(string & s, string & p, int i, int j, vector<vector<int>>& dp) {
	if (j >= p.size()) {
		return i >= s.size();
	}
	if (dp[i][j] != 0)return dp[i][j] == 1 ? true : false;
	else {
		bool p1=false;
		if (p[j] == '*') {
			for (int l = 0; l <= s.size() - i; l++) {
				p1 = p1 || isMatch_dfs(s, p, l + i, j + 1, dp);
			}
		}
		else if (p[j] == '?') {
			p1 = (i < s.size()) && isMatch_dfs(s, p, i + 1, j + 1, dp);
		}
		else {
			p1 = (s[i] == p[j]) && isMatch_dfs(s, p, i + 1, j + 1, dp);
		}
		dp[i][j] = p1 == true ? 1 : 2;
		return p1;
	}
}
bool isMatch2(string s, string p) {
	vector<vector<int>> dp(s.size() + 1, vector<int>(p.size() + 1, 0));
	return isMatch_dfs(s, p, 0, 0, dp);
}

//top面试 T46
void permute_dfs(vector<int>& nums, vector<vector<int>>& res, vector<int>& path, vector<bool>& used) {
	if (path.size() >= nums.size()) {
		res.push_back(path);
		return;
	}
	//把每一个没有用过的数 用一下
	for (int i = 0; i < nums.size(); i++) {
		if (!used[i]) {
			path.push_back(nums[i]);
			used[i] = true;
			permute_dfs(nums, res, path, used);
			used[i] = false;
			path.pop_back();
		}
	}
}
vector<vector<int>> permute(vector<int>& nums) {
	vector<vector<int>> res;
	vector<int> path;
	vector<bool> used(nums.size(), false);
	permute_dfs(nums, res, path, used);
	return res;
}

//top面试 T48
void rotate(vector<vector<int>>& matrix) {
	//主对角线对称
	for (int i = 0; i < matrix.size(); i++) {
		for (int j = 0; j < matrix[0].size(); j++) {
			if (i < j)swap(matrix[i][j], matrix[j][i]);
		}
	}
	//中间列对称
	for (int i = 0; i < matrix.size(); i++) {
		for (int j = 0; j < matrix[0].size(); j++) {
			if (j < matrix[0].size() / 2)swap(matrix[i][j], matrix[i][matrix[0].size() - 1 - j]);
		}
	}
}

//top面试 T49
vector<vector<string>> groupAnagrams(vector<string>& strs) {
	vector<vector<string>> res;
	//存相同字母字符串的下标
	map<string, vector<int>> m;
	for (int i = 0; i < strs.size(); i++) {
		string t = strs[i];
		sort(t.begin(), t.end());
		m[t].push_back(i);
	}
	for (auto [key, value] : m) {
		vector<string> path;
		for (auto i : value) {
			path.push_back(strs[i]);
		}
		res.push_back(path);
	}
	return res;
}

//top面试 T50
double myPow(double x, int n) {
	if (n == 0 || x == 1)return 1;
	double a = x;
	int b = n;
	//如果n是最小值 把n＋1 这样取绝对值才是有效的
	if (n == INT_MIN)b++;
	double ans = 1;
	bool sign = n > 0;
	b = abs(b);
	while (b) {
		if (b & 1)ans = ans * a;
		a = a * a;
		b = b >> 1;
	}
	//如果n是最小值 由于前面＋1 所以需要最后在乘一个x
	if (n == INT_MIN)ans = ans * x;
	return sign ? ans : 1 / ans;
}

//二叉树 递归
int sumRootToLeaf_dfs(Tnode* root, int res) {
	if (root == nullptr)return 0;
	//本次逻辑 累加值左移一位 加上当前值
	res = (res << 1) | root->val;
	//如果是叶子结点 直接返回
	if (root->left == nullptr&&root->right == nullptr)return res;
	//否则 继续递归下去
	return sumRootToLeaf_dfs(root->left, res) + sumRootToLeaf_dfs(root->right, res);
}
int sumRootToLeaf(Tnode* root) {
	int res = sumRootToLeaf_dfs(root, 0);
	return res;
}

//top面试 T54
vector<int> spiralOrder(vector<vector<int>>& matrix) {
	//当前 行 列
	int i = 0, j = 0;
	//开始 行 列
	int sr = 0, sc = 0;
	//结束 行 列
	int m = matrix.size(), n = matrix[0].size();
	vector<int> path;
	while (1) {
		if (path.size() == matrix.size()*matrix[0].size())break;
		for (j; j < n; j++) { path.push_back(matrix[i][j]); }
		//找过上面一行后 当前行++，当前列-- 开始行++
		i++;
		j--;
		sr++;
		if (path.size() == matrix.size()*matrix[0].size())break;
		for (i; i < m; i++) { path.push_back(matrix[i][j]); }
		//找过后面一列后 当前列--，当前行-- 结束列--
		j--;
		i--;
		n--;
		if (path.size() == matrix.size()*matrix[0].size())break;
		for (j; j >= sc; j--) { path.push_back(matrix[i][j]); }
		//找过下面一行后 当前行--，当前列++ 结束行--
		i--;
		j++;
		m--;
		if (path.size() == matrix.size()*matrix[0].size())break;
		for (i; i >= sr; i--) { path.push_back(matrix[i][j]); }
		//找过前面一列后 当前列++，当前行++ 开始列++
		j++;
		i++;
		sc++;
	}
	return path;
}

//top面试 T54
bool canJump(vector<int>& nums) {
	//维持一个可以到达的最远距离
	int maxdist = 0;
	for (int i = 0; i < nums.size(); i++) {
		//当前值在可以到达的最远距离内的时候，可能会更新最远距离
		if (i < maxdist)maxdist = max(maxdist, nums[i] + i);
		if (maxdist >= nums.size()-1)return true;
	}
	return false;
}

//top面试 T56
vector<vector<int>> merge(vector<vector<int>>& intervals) {
	vector<vector<int>> res;
	vector<int> path(2);
	//默认 就是对第一个元素的升序排列
	sort(intervals.begin(), intervals.end());
	path = intervals[0];
	for (int i = 1; i < intervals.size(); i++) {
		/*
		如果当前区间的左端点在数组 merged 中最后一个区间的右端点之后，那么它们不会重合，
		    我们可以直接将这个区间加入数组 merged 的末尾；
		否则，它们重合，我们需要用当前区间的右端点更新数组 merged 中最后一个区间的右端点，
		    将其置为二者的较大值。
		*/
		if (intervals[i][0] > path[1]) {
			res.push_back(path);
			path = intervals[i];
		}
		else {
			path[1] = max(path[1], intervals[i][1]);
		}
	}
	res.push_back(path);
	return res;
}

//top面试 T62
int uniquePaths_dfs(int m, int n, int i, int j) {
	if (m == i && n == j) {
		return 1;
	}
	if (i > m || j > n)return 0;
	int p1 = uniquePaths_dfs(m, n, i + 1, j);
	int p2 = uniquePaths_dfs(m, n, i, j + 1);
	return p1 + p2;
}
int uniquePaths(int m, int n) {
	return uniquePaths_dfs(m, n, 0, 0);
}
int uniquePathsdp(int m, int n) {
	vector<vector<long>> dp(m + 1, vector<long>(n + 1));
	for (int i = 0; i <= m; i++)dp[i][n] = 1;
	for (int i = 0; i <= n; i++)dp[m][i] = 1;
	//从右下向左上遍历
	for (int i = m - 1; i >= 0; i--) {
		for (int j = n - 1; j >= 0; j--) {
			dp[i][j] = dp[i + 1][j] + dp[i][j + 1];
		}
	}
	return ((int)dp[1][1]);
}

//top面试 T66
vector<int> plusOne(vector<int>& digits) {
	//从后向前找第一个不是9的数字 把他++
	for (int i = digits.size() - 1; i >= 0; i--) {
		if (digits[i] != 9) {
			digits[i]++;
			return digits;
		}
		else {
			digits[i] = 0;
		}
	}
	//没有不是9的数字 在前面插 1
	digits.insert(digits.begin(), 1);
	return digits;
}

//top面试 T69
int mySqrt(int x) {
	long l = 0;
	long r = (long)x + 1;
	while ((l + 1) != r) {
		long mid = l + (r - l) / 2;
		if (mid*mid <= x)l = mid;
		else r = mid;
	}
	return (int)l;
}

//top面试 T70
int climbStairs(int n) {
	vector<int> dp(n + 1, 0);
	dp[n - 1] = 1;
	if (n >= 2)dp[n - 2] = 2;
	for (int i = n - 3; i >= 0; i--) {
		dp[i] = dp[i + 1] + dp[i + 2];
	}
	return dp[0];
}

//top面试 T73
void setZeroes(vector<vector<int>>& matrix) {
	vector<bool> r(matrix.size());//某一行是否要改为0
	vector<bool> c(matrix[0].size());//某一列是否要改为0
	for (int i = 0; i < matrix.size(); i++) {
		for (int j = 0; j < matrix[0].size(); j++) {
			if (matrix[i][j] == 0) {
				r[i] = true;
				c[j] = true;
			}
		}
	}
	for (int i = 0; i < r.size(); i++) {
		if (r[i]) {
			for (int j = 0; j < matrix[0].size(); j++)matrix[i][j] = 0;
		}
	}
	for (int i = 0; i < c.size(); i++) {
		if (c[i]) {
			for (int j = 0; j < matrix.size(); j++)matrix[j][i] = 0;
		}
	}
	return;
}

//top面试 T75
void sortColors(vector<int>& nums) {
	// priority_queue<int,vector<int>,greater<int>> m;
	// for(int i:nums)m.push(i);
	// int i = 0;
	// while(!m.empty()){
	//     nums[i] = m.top();
	//     m.pop();
	//     i++;
	// }
	//三色国旗 l 小于1的右边界 r 大于1的左边界
	int l = -1, r = nums.size();
	int c = 0;
	while (c < r) {
		//右边界右移，进行下一个元素
		if (nums[c] < 1) {
			l++;
			swap(nums[l], nums[c]);
			c++;
		}
		//进行下一个元素
		else if (nums[c] == 1) {
			c++;
		}
		//左边界左移
		else {
			r--;
			swap(nums[c], nums[r]);
		}
	}
}

//top面试 T76
string minWindow(string s, string t) {
	unordered_map<char, int> m;
	for (char c : t)m[c]++;
	int tot = t.size();
	int res = INT_MAX;//满足条件的子串的长度
	int l = 0, r = 0;
	int st = 0;//满足条件的子串的起始位置
	while (l <= r && r < s.size()) {
		//右扩
		while (r < s.size() && tot>0) {
			if (m.count(s[r])) {
				m[s[r]]--;
				//如果减了之后还大于0 说明是有效的 tot要--
				if (m[s[r]] >= 0)tot--;
			}
			r++;
		}
		//更新答案，左边界右扩
		while (tot == 0) {
			if (res > r - l) {
				res = r - l;
				st = l;
			}
			if (m.count(s[l])) {
				m[s[l]]++;
				//如果加了之后大于0 说明此时[l,r]之内已经不满足要求了 tot++
				if (m[s[l]] > 0)tot++;
			}
			l++;
		}
	}
	if (res == INT_MAX)return "";
	return s.substr(st, res);
}

//top面试 T78
void subsets_dfs(vector<int>& nums, vector<vector<int>>& res, vector<int>& path, int len, int index) {
	if (path.size() == len) {
		res.push_back(path);
		return;
	}
	//从index开始寻找长度为len的子集
	for (int i = index; i < nums.size(); i++) {
		path.push_back(nums[i]);
		/*
		index++ 递归过程中index一直变大
		如果 subsets_dfs(nums, res, path, len, index+1); 则index在递归过程中有回溯，会变小 是错误的
		*/
		index++;
		subsets_dfs(nums, res, path, len, index);
		path.pop_back();
	}
}
vector<vector<int>> subsets(vector<int>& nums) {
	vector<vector<int>> res;
	vector<int> path;
	//对每一个长度的子集做递归
	for (int i = 0; i <= nums.size(); i++)subsets_dfs(nums, res, path, i, 0);
	return res;
}

//top面试 T79  单词搜索
bool exist_dfs(vector<vector<char>>& board, string& word, int i, int j, int index) {
	//每个字符都匹配了，返回匹配
	if (index == word.size()) {
		return true;
	}
	bool p1 = false, p2 = false, p3 = false, p4 = false;
	//如果当前字符相等 往4个方向递归继续匹配
	if (board[i][j] == word[index]) {
		char c = board[i][j];
		board[i][j] = '0';
		index++;
		if (i + 1 < board.size())p1 = exist_dfs(board, word, i + 1, j, index);
		if (j + 1 < board[0].size())p2 = exist_dfs(board, word, i, j + 1, index);
		if (i - 1 >= 0)p3 = exist_dfs(board, word, i - 1, j, index);
		if (j - 1 >= 0)p4 = exist_dfs(board, word, i, j - 1, index);
		index--;
		board[i][j] = c;
	}
	//否则 返回不匹配
	return p1 || p2 || p3 || p4;
}
bool exist(vector<vector<char>>& board, string word) {
	if (board.size() == 1 && board[0].size() == 1 && word.size() == 1 && board[0][0] == word[0])return true;
	for (int i = 0; i < board.size(); i++) {
		for (int j = 0; j < board[0].size(); j++) {
			//把每一个位置当作起点 寻找
			if (exist_dfs(board, word, i, j, 0))return true;
		}
	}
	return false;
}

//单词搜索2 212
struct Trienode {
	string word;
	vector<Trienode*> next;
	Trienode() {
		this->word = "";
		next = vector<Trienode*>(26, nullptr);
	}
};

class Solution212 {
public:
	void inserttrie(Trienode* root, string str) {
		Trienode* node = root;
		for (char c : str) {
			if (node->next[c - 'a'] == nullptr) {
				node->next[c - 'a'] = new Trienode;
			}
			node = node->next[c - 'a'];
		}
		node->word = str;
	}
	int dir[4][2] = { {1, 0}, {-1, 0}, {0, 1}, {0, -1} };
	void exist_dfs(vector<vector<char>>& board, int m, int n, set<string>& Set, int i, int j, Trienode* root) {
		char cur = board[i][j];
		//当前字母不在字典树中
		if (root->next[cur - 'a'] == nullptr)return ;
		root = root->next[cur - 'a'];
		//当前字母在字典树中，且存在以当前字母为结尾的单词
		if (root->word.size() > 0) {
			Set.insert(root->word);
		}
		board[i][j] = '0';
		for (int d = 0; d < 4; d++) {
			int mx = i + dir[d][0];
			int my = j + dir[d][1];
			if (mx >= 0 && mx < m&&my >= 0 && my < n&&board[mx][my] != '0')exist_dfs(board, m, n, Set, mx, my, root);
		}
		board[i][j] = cur;
		return;
	}
	vector<string> findWords(vector<vector<char>>& board, vector<string>& words) {
		vector<string> ans;
		set<string> Set;
		Trienode* root = new Trienode;
		int m = board.size(), n = board[0].size();
		//构建字典树
		for (string str : words) {
			inserttrie(root, str);
		}
		//每一个位置作为起点搜索
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				exist_dfs(board, m, n, Set, i, j, root);
			}
		}
		for (auto str : Set)ans.push_back(str);
		return ans;
	}
};
//top面试 T84  找小的元素 单调增栈
int largestRectangleArea(vector<int>& heights) {
	//实际上问题 寻找每个元素左边右边第一个比他小的元素
	stack<int> s;
	int res = INT_MIN;
	//当前面积
	int cur;
	for (int i = 0; i < heights.size(); i++) {
		//当前元素比栈顶元素大 可以入栈
		if (s.empty() || heights[s.top()] < heights[i]) {
			s.push(i);
		}
		//当前元素小于栈顶元素 出栈 出栈时收集更新结果
		while (!s.empty() && heights[s.top()] > heights[i]) {
			int curh = heights[s.top()];
			s.pop();
			//当一个元素出栈后  
			//栈顶元素为 出栈元素的的左边比他小的元素
			//for循环当前遍历到的元素为 出栈元素的右边比他小的元素
			//如果栈为空 当前元素的左边没有比他小的元素了
			if (s.empty())cur = curh * i;
			else cur = curh * (i - s.top() - 1);
			res = max(res, cur);
		}
		s.push(i);
	}
	//此时所有元素右边都没有更小的了
	while (!s.empty()) {
		int curh = heights[s.top()];
		s.pop();
		//如果栈为空 当前元素的左边没有比他小的元素了
		if (s.empty())cur = curh * heights.size();
		//否则 栈顶元素为 出栈元素的的左边比他小的元素
		else cur = curh * (heights.size() - s.top() - 1);
		res = max(res, cur);
	}
	return res;
}

//top面试 T88
void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
	vector<int> t;
	int p1 = 0, p2 = 0;
	while (p1 < m&&p2 < n) {
		if (nums1[p1] < nums2[p2]) {
			t.push_back(nums1[p1]);
			p1++;
		}
		else {
			t.push_back(nums2[p2]);
			p2++;
		}
	}
	while (p1 < m) {
		t.push_back(nums1[p1]);
		p1++;
	}
	while (p2 < n) {
		t.push_back(nums2[p2]);
		p2++;
	}
	nums1 = t;
}

//top面试 T91
long numDecodings_dfs(string& s, int index, vector<long>& dp) {
	//先查表
	if (dp[index] != -1)return dp[index];
	if (index == s.size())return 1;
	//-1 无效
	if (s[index] == '0')return -1;

	long p1 = 0, p2 = 0;
	p1 = numDecodings_dfs(s, index + 1, dp);
	if (p1 == -1)p1 = 0;
	if ((s[index] == '1'&&index + 1 < s.size()) || (s[index] == '2'&&index + 1 < s.size() && s[index + 1] <= '6')) {
		p2 = numDecodings_dfs(s, index + 2, dp);
		if (p2 == -1)p2 = 0;
	}
	dp[index] = p1 + p2;
	return p1 + p2;
}
int numDecodings(string s) {
	vector<long> dp(s.size() + 1, -1);
	long c = numDecodings_dfs(s, 0, dp);
	if (c == -1)return 0;
	return (int)c;
}
int numDecodingsdp(string s) {
	//设 dp[i] 表示字符串 s的前 i 个字符 的解码方法数
	vector<int> dp(s.size() + 1);
	dp[0] = 1;
	for (int i = 1; i <= s.size(); i++) {
		if (s[i - 1] != '0')dp[i] = dp[i] + dp[i - 1];
		if (i > 1 && s[i - 2] != '0' && ((s[i - 2] - '0') * 10 + (s[i - 1] - '0') <= 26)) {
			dp[i] = dp[i] + dp[i - 2];
		}
	}
	return dp[s.size()];
}

//左右孩子节点都不为空，则将删除节点的左子树放到删除节点的右子树的最左节点的左孩子的位置
// 并返回删除节点右孩子为新的根节点。
Tnode* deleteNode(Tnode* root, int key) {
	if (root == nullptr) return root; // 第一种情况：没找到删除的节点，遍历到空节点直接返回了
	if (root->val == key) {
		// 第二种情况：左右孩子都为空（叶子节点），直接删除节点， 返回NULL为根节点
		if (root->left == nullptr && root->right == nullptr) {
			return nullptr;
		}
		// 第三种情况：其左孩子为空，右孩子不为空，删除节点，右孩子补位 ，返回右孩子为根节点
		else if (root->left == nullptr) {
			root = root->right;
			return root;
		}
		// 第四种情况：其右孩子为空，左孩子不为空，删除节点，左孩子补位，返回左孩子为根节点
		else if (root->right == nullptr) {
			root = root->left;
			return root;
		}
		// 第五种情况：左右孩子节点都不为空，则将删除节点的左子树放到删除节点的右子树的最左面节点的左孩子的位置
		// 并返回删除节点右孩子为新的根节点。
		else {
			Tnode* cur = root->right; // 找右子树最左面的节点
			while (cur->left != nullptr) {
				cur = cur->left;
			}
			cur->left = root->left; // 把要删除的节点（root）左子树放在cur的左孩子的位置
			root = root->right;     // 返回旧root的右孩子作为新root
			return root;
		}
	}
	if (root->val > key) root->left = deleteNode(root->left, key);
	if (root->val < key) root->right = deleteNode(root->right, key);
	return root;
}

//top面试 T98
bool isValidBST_h(Tnode* root, long l, long r) {
	if (root == nullptr)return true;

	if (root->val >= r || root->val <= l)return false;
	//向左子树递归时 上边界更新为当前结点值
	bool p1 = isValidBST_h(root->left, l, root->val);
	//向右子树递归时 下边界更新为当前结点值
	bool p2 = isValidBST_h(root->right, root->val, r);
	return p1 && p2;
}
bool isValidBST(Tnode* root) {
	//随着递归深度加深，结点数字的边界缩小
	return isValidBST_h(root, LONG_MIN, LONG_MAX);
}

int consecutiveNumbersSum(int n) {
	int l = 1, r = 1;
	int res = 1;
	int tmp = 0;
	while (r < n) {
		while (tmp < n&&r < n) {
			tmp = tmp + r;
			r++;
		}
		if (tmp == n) {
			res++;
			tmp = tmp - l;
			l++;
		}
		while (tmp > n) {
			tmp = tmp - l;
			l++;
		}
	}
	return res;
}
/*
利用求和公式 
我们已知和为 n, 那么就要求上述方程的正整数解，但有两个未知数 a,k，
一种简单的思路是：枚举其中一个数，那么另外一个数字就能计算出来了，
我们只需验证得到的解是否合法即可，这里的合法是指解出的 a,k 应该是正整数。
如果由某个k得到的首项a小于1，则结束循环
*/
int consecutiveNumbersSum2(int n) {
	int res = 1;
	for (int k = 2;; k++) {
		int a = (2 * n / k + 1 - k) / 2;
		if (a < 1)break;
		if (k*(2 * a + k - 1) / 2 == n)res++;
	}
	return res;
}

//top面试 T101
bool isSymmetric_h(Tnode* root1, Tnode* root2) {
	if (root1 == nullptr&&root2 == nullptr)return true;
	if ((root1 == nullptr&&root2 != nullptr) || (root2 == nullptr&&root1 != nullptr))return false;
	if ((root1->val != root2->val))return false;
	return isSymmetric_h(root1->left, root2->right) && isSymmetric_h(root1->right, root2->left);
}
bool isSymmetric(Tnode* root) {
	return isSymmetric_h(root->left, root->right);
}

//top面试 T102
vector<vector<int>> levelOrder(Tnode* root) {
	vector<vector<int>> res;
	queue<Tnode* >q;
	if (root)q.push(root);
	while (!q.empty()) {
		int len = q.size();
		vector<int> path;
		for (int i = 0; i < len; i++) {
			Tnode* node = q.front();
			q.pop();
			path.push_back(node->val);
			if (node->left)q.push(node->left);
			if (node->right)q.push(node->right);
		}
		res.push_back(path);
	}
	return res;
}

//top面试 T104
int maxDepth(Tnode* root) {
	if (root == nullptr)return 0;
	return 1 + max(maxDepth(root->left), maxDepth(root->right));
}

//top面试 T105   关键 找根结点 分割数组
Tnode* bulidT(vector<int>& inorder, int il, int ir, vector<int>& postorder, int pl, int pr) {
	//根据中序和后续建树，没有重复元素
	//输入 中序的起始和后序的起始 注意：左闭右开
	if (pl == pr)return nullptr;
	//从后序中找根结点
	Tnode* root = new Tnode(postorder[pr - 1]);
	if (pr - pl == 1)return root;
	//找中序数组分割点
	int index;
	for (index = il; index < ir; index++) {
		if (inorder[index] == root->val)break;
	}
	//切割中序数组
	int leftInorderSt = il;
	int leftInorderEd = index;
	int rightInorderSt = index + 1;
	int rightInorderEd = ir;
	//切割后续数组
	int leftPosorderSt = pl;
	int leftPosorderEd = pl + index - il;
	int rightPosorderSt = leftPosorderEd;
	int rightPosorderEd = pr - 1;
	root->left = bulidT(inorder, leftInorderSt, leftInorderEd, postorder, leftPosorderSt, leftPosorderEd);
	root->right = bulidT(inorder, rightInorderSt, rightInorderEd, postorder, rightPosorderSt, rightPosorderEd);
	return root;

}

Tnode* bulidT2(vector<int>& inorder, int il, int ir, vector<int>& pre, int pl, int pr) {
	//根据中序和前序建树，没有重复元素
	//输入 中序的起始和前序的起始 注意：左闭右开
	if (pl == pr)return nullptr;
	Tnode* root = new Tnode(pre[pl]);
	if (pr - pl == 1)return root;
	int index = 0;
	for (index = il; index < ir; index++) {
		if (inorder[index] == root->val)break;
	}
	//分割前序数组
	int leftPreSt = pl + 1;
	int leftPreEd = leftPreSt + index - il;
	int rightPreSt = leftPreEd;
	int rightPreEd = pr;
	//分割中序数组
	int leftInorSt = il;
	int leftInorEd = index;
	int rightInorSt = index + 1;
	int rightInorEd = ir;
	root->left = bulidT2(inorder, leftInorSt, leftInorEd, pre, leftPreSt, leftPreEd);
	root->right = bulidT2(inorder, rightInorSt, rightInorEd, pre, rightPreSt, rightPreEd);
	return root;
}

//108 有序数组转为二叉搜索树
Tnode* sortedArrayToBST_h(vector<int>& nums, int l, int r) {
	if (l == r)return new Tnode(nums[l]);
	int mid = (l + r) / 2;
	Tnode * node = new Tnode(nums[mid]);
	if (mid - 1 >= l)node->left = sortedArrayToBST_h(nums, l, mid - 1);
	if (mid + 1 <= r)node->right = sortedArrayToBST_h(nums, mid + 1, r);
	return node;

}
Tnode* sortedArrayToBST(vector<int>& nums) {
	return sortedArrayToBST_h(nums, 0, nums.size() - 1);
}

//121 买卖股票的最佳时机
int maxProfit(vector<int>& prices) {
	int premin = prices[0];
	int ans = INT_MIN;
	for (int i = 1; i < prices.size(); i++) {
		premin = min(premin, prices[i]);
		ans = max(ans, prices[i] - premin);
	}
	return ans == INT_MIN ? 0 : ans;
}

//122 买卖股票的最佳时机2
int maxProfit2(vector<int>& prices) {
	int ans = 0;
	int indate;
	bool hold = false;
	for (int i = 0; i < prices.size() - 1; i++) {
		if (!hold) {
			if (prices[i + 1] > prices[i]) {
				indate = i;
				hold = true;
			}
		}
		else {
			if (prices[i + 1] < prices[i]) {
				ans = ans + prices[i] - prices[indate];
				hold = false;
			}
		}
	}
	if (hold)ans = ans + prices.back() - prices[indate];
	return ans;
}
int maxProfitdp(vector<int>& prices) {
	int n = prices.size();
	vector<vector<int>> dp(n, vector<int>(2));
	dp[0][0] = 0, dp[0][1] = -prices[0];
	for (int i = 1; i < n; ++i) {
		dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
		dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
	}
	return dp[n - 1][0];
}

//124 二叉树的最大路径和
/*
首先，考虑实现一个简化的函数 maxGain(node)，该函数计算二叉树中的一个节点的最大贡献值，
具体而言，就是在以该节点为根节点的子树中寻找以该节点为起点的一条路径，使得该路径上的节点值之和最大。
具体而言，该函数的计算如下。
	空节点的最大贡献值等于 0。
	非空节点的最大贡献值等于节点值与其子节点中的最大贡献值之和（对于叶节点而言，最大贡献值等于节点值）。
得到每个节点的最大贡献值之后，
对于二叉树中的一个节点，该节点的最大路径和取决于该节点的值与该节点的左右子节点的最大贡献值，
如果子节点的最大贡献值为正，则计入该节点的最大路径和，否则不计入该节点的最大路径和。
维护一个全局变量 maxSum 存储最大路径和，在递归过程中更新 maxSum 的值，
最后得到的 maxSum 的值即为二叉树中的最大路径和。
*/
int maxPathSum_h(Tnode* root, int& val) {
	if (root == nullptr) {
		return 0;
	}
	//如果是负数 记为0
	int p1 = max(0, maxPathSum_h(root->left, val));
	int p2 = max(0, maxPathSum_h(root->right, val));
	//当前节点的最大路径和
	int p = root->val + p1 + p2;
	//找到以每一个节点为起点的最大的路径和
	val = max(val, p);
	//当前节点的最大贡献值 左右分支只能走一个 选最大的
	return root->val + max(p1, p2);
}
int maxPathSum(Tnode* root) {
	int sum = INT_MIN;
	maxPathSum_h(root, sum);
	return sum;
}

//127 单词接龙
int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
	if (wordList.end() == find(wordList.begin(), wordList.end(), endWord))return 0;
	unordered_map<string, int> m;
	for (string str : wordList)m[str]++;
	queue<string> q;
	int step = 1;
	q.push(beginWord);
	while (!q.empty()) {
		int len = q.size();
		for (int i = 0; i < len; i++) {
			string tmp = q.front();
			q.pop();
			for (int j = 0; j < tmp.size(); j++) {
				string tmp2 = tmp;
				for (int k = 0; k < 26; k++) {
					tmp2[j] = 'a' + k;
					if (tmp2 != tmp && m.count(tmp2) && m[tmp2] != 0) {
						if (tmp2 == endWord)return step + 1;
						q.push(tmp2);
						m[tmp2] = 0;
					}
				}
			}
		}
		step++;
	}
	return 0;
}

//128 最长连续序列
int longestConsecutive(vector<int>& nums) {
	//去重
	unordered_set<int> s;
	for (int num : nums)s.insert(num);
	int res = 0;
	for (int num : s) {
		//对于每一个元素 只有当他的前一个元素不存在的时候才开始寻找后面连续的元素
		if (!s.count(num - 1)) {
			int cur = num;
			int curlength = 1;
			while (s.count(cur + 1)) {
				cur++;
				curlength++;
			}
			res = max(res, curlength);
		}
	}
	return res;
}

//130 被围绕的区域
/*
	任何边界上的 O 都不会被填充为 X。 找到不会被包围的o，剩下的都是会被包围的o
	所有的不被包围的 O 都直接或间接与边界上的 O 相连。
*/
void solve(vector<vector<char>>& board) {
	int m = board.size(), n = board[0].size();
	vector<int> dx = { 0,0,1,-1 };
	vector<int> dy = { 1,-1,0,0 };
	queue<pair<int, int>> q;
	//找到四周的o 并把其换位a
	for (int i = 0; i < m; i++) {
		if (board[i][0] == 'O') {
			q.emplace(i, 0);
			board[i][0] = 'a';
		}
		if (board[i][n - 1] == 'O') {
			q.emplace(i, n - 1);
			board[i][n - 1] = 'a';
		}
	}
	for (int i = 0; i < n; i++) {
		if (board[0][i] == 'O') {
			q.emplace(0, i);
			board[0][i] = 'a';
		}
		if (board[m - 1][i] == 'O') {
			q.emplace(m - 1, i);
			board[m - 1][i] = 'a';
		}
	}
	//寻找和边界上的o相连的o 并标记为a
	while (!q.empty()) {
		int x = q.front().first, y = q.front().second;
		q.pop();
		for (int i = 0; i < 4; i++) {
			int mx = x + dx[i], my = y + dy[i];
			if (mx < 0 || my < 0 || mx >= m || my >= n || board[mx][my] != 'O')continue;
			q.emplace(mx, my);
			board[mx][my] = 'a';
		}
	}
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (board[i][j] == 'a')board[i][j] = 'O';
			else if (board[i][j] == 'O')board[i][j] = 'X';
		}
	}
}

//131 分割回文串
void tpartition_dfs(vector<vector<string>>& res, vector<string>& path, string s, int index, vector<vector<bool>>& dp) {
	if (index >= s.size()) {
		res.push_back(path);
		return;
	}
	//对于起点index 枚举在这个起点可能构成回文串的终点 i
	for (int i = index; i < s.size(); i++) {
		if (dp[index][i]) {
			path.push_back(s.substr(index, i - index + 1));
			tpartition_dfs(res, path, s, i + 1, dp);
			path.pop_back();
		}
	}
}
vector<vector<string>> tpartition(string s) {
	vector<vector<string>> res;
	vector<string> path;
	//预处理s[i,j]是否是回文串
	// j>=i 只有一个字符或者空串为回文串
	// 否则 dp[i][j] = dp[i + 1][j - 1] && s[i] == s[j];
	//dp[i][j] 依赖 dp[i+1][j-1] 所以i从大到小遍历，j从小到大遍历
	vector<vector<bool>> dp(s.size(), vector<bool>(s.size(), true));
	for (int i = s.size() - 1; i >= 0; i--) {
		for (int j = i + 1; j < s.size(); j++) {
			dp[i][j] = dp[i + 1][j - 1] && s[i] == s[j];
		}
	}
	tpartition_dfs(res, path, s, 0, dp);
	return res;
}

bool isok(string s, int i, int j) {
	//判断从i 开始长度为j的字符串是不是回文
	if (j == 1)return true;
	j = j + i - 1;
	while (i < j) {
		if (s[i] != s[j])return false;
		i++;
		j--;
	}
	return true;
}
void partition_dfs(vector<vector<string>>& res, vector<string>& path, string s, int index) {
	if (index >= s.size()) {
		res.push_back(path);
		return;
	}
	//从index开始长度为i的字符串是否是回文串
	for (int i = 1; i <= s.size() - index; i++) {
		if (isok(s, index, i)) {
			path.push_back(s.substr(index, i));
			partition_dfs(res, path, s, index + i);
			path.pop_back();
		}
	}
}
vector<vector<string>> partition2(string s) {
	vector<vector<string>> res;
	vector<string> path;
	partition_dfs(res, path, s, 0);
	return res;
}

//134 加油站
int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
	int res = 0;
	int run = 0;
	int st = 0;
	for (int i = 0; i < gas.size(); i++) {
		// 一个站的收益如果小于0，肯定不能作为起点；
		//而连续的多个站也可以等效地看做一个站，如果其累积收益小于0，就跳过，寻找下一个。
		run = run + gas[i] - cost[i];
		//总和必须大于等于0，否则不能完成绕行
		res = res + gas[i] - cost[i];
		if (run < 0) {
			st = i + 1;
			// 还原到初始状态
			run = 0;
		}

	}
	return res < 0 ? -1 : st;

}

//136 出现一次的数字 任何数字和0异或结果都是本身
int singleNumber(vector<int>& nums) {
	int res = 0;
	for (int num : nums) {
		res = res ^ num;
	}
	return res;
}
//两个相同的数字异或结果是0，任何数字和0异或结果都是本身
vector<int> t12(vector<int>& nums) {
	//数组中只有两个数字是出现一次的，别的数字都是出现了两次
	//返回那两个数字
	int tmp = 0;
	for (int i = 0; i < nums.size(); i++) {
		tmp = tmp ^ nums[i];
	}
	//tmp是两个只出现一次的数字的异或
	//找到tmp二进制中的一个1 记录这个1的位置
	//然后对数组中的所有数字按照这个这个位置的数值分为两组
	//则 两个出现一次的数字肯定会被分在不同组中
	//tmp&(1<<i) 找tmp中1的位置 
	int indx = 0;
	for (int i = 0; i < 32; i++) {
		if (tmp&(1 << i)) {
			indx = i;
			break;
		}
	}
	//对所有数字分组
	vector<int> num1;
	vector<int> num2;
	for (int i = 0; i < nums.size(); i++) {
		if (nums[i] & (1 << indx))num1.push_back(nums[i]);
		else num2.push_back(nums[i]);
	}
	//对两个数组内的数字分别异或
	int a = 0, b = 0;
	for (int i = 0; i < num1.size(); i++) {
		a = a ^ num1[i];
	}
	for (int i = 0; i < num2.size(); i++) {
		b = b ^ num2[i];
	}
	return { a,b };
}
//所有数字都出现3次 有一个出现一次，找到出现一次的数字
int t13(vector<int>& nums) {
	//统计每个位置出现1的次数
	vector<int> count(32, 0);
	for (int i = 0; i < 32; i++) {
		for (int j = 0; j < nums.size(); j++) {
			count[i] = count[i] + (nums[j] & (1 << i));
		}
	}
	//如果某个位置出现1的次数不是3的倍数，则出现一次的那个数字的这个位置一定是1，
	//找到出现一次的数字的所有1的位置，拼接出只出现一次的数字

	int res = 0;
	for (int i = 0; i < 32; i++) {
		if (count[i] % 3 != 0) {
			res = res + (1 << i);
		}
	}
	return res;
}

//138 复制带随机指针的链表
RNode* copyRandomList(RNode* head) {
	if (head == NULL)return NULL;
	unordered_map<RNode*, RNode*> m;
	RNode* root = head;
	while (head) {
		m[head] = new RNode(head->val);
		head = head->next;
	}
	head = root;
	while (head) {
		m[head]->next = m[head->next];
		m[head]->random = m[head->random];
		head = head->next;
	}
	return m[root];
}

//139 单词拆分
bool wordBreak_dfs(string& s, unordered_set<string>& wordSet, int index, vector<int>& dp) {
	if (index >= s.size())return true;
	if (dp[index] != -1)return dp[index];
	//从index开始截取每一个长度，判断截取下的字符串是否在单词表中
	for (int i = index; i < s.size(); i++) {
		string word = s.substr(index, i - index + 1);
		if (wordSet.find(word) != wordSet.end() && wordBreak_dfs(s, wordSet, i + 1, dp)) {
			dp[index] = 1;//表示可以被拆分
			return true;
		}
	}
	dp[index] = 0;//表示不可以被拆分
	return false;
}
bool wordBreak(string s, vector<string>& wordDict) {
	//-1，表示可以初始
	vector<int> dp(s.size(), -1);
	unordered_set<string> wordSet(wordDict.begin(), wordDict.end());
	return wordBreak_dfs(s, wordSet, 0, dp);
}
bool wordBreakdp(string s, vector<string>& wordDict) {
	unordered_set<string> wordSet(wordDict.begin(), wordDict.end());
	vector<bool> dp(s.size() + 1, false);
	//dp[i] : 字符串长度为i的话，dp[i]为true，表示可以拆分为一个或多个在字典中出现的单词
	dp[0] = true;
	for (int i = 1; i <= s.size(); i++) {
		for (int j = 0; j < i; j++) {
			//从j开始截取长度为i-j的字符串
			string str = s.substr(j, i - j);
			//如果这个单词在单词表中，且字符串长度为 j 的可以拆分为一个或多个在字典中的单词
			//则 字符串长度为 i 的可以拆分为一个或多个在字典中的单词
			if (wordSet.find(str) != wordSet.end() && dp[j])dp[i] = true;
		}
	}
	return dp.back();
}

//140 单词拆分
void wordBreak_dfs2(string s, unordered_map<int, vector<string>>& m,unordered_set<string>& wordSet, int index) {
	//从index开始可以组成的句子列表还是未知的 则往下进行
	if (!m.count(index)) {
		if (index >= s.size()) {
			m[index] = { "" };
			return;
		}
		m[index] = {};
		// i 截取的字符在原始字符串中的结束位置下标
		for (int i = index + 1; i <= s.size(); i++) {
			string str = s.substr(index, i - index);
			//截取到的字符存在与字典中时，递归
			if (wordSet.count(str)) {
				wordBreak_dfs2(s, m, wordSet, i);
				//当前字符串加上 i 后面的组成的句子，构成当前位置组成的句子
				for (auto stemp : m[i]) {
					m[index].push_back(stemp.empty() ? str : str + " " + stemp);
				}
			}

		}
	}
}
vector<string> wordBreak2(string s, vector<string>& wordDict) {
	//从每个下标开始可以组成的句子列表
	unordered_map<int, vector<string>> m;
	unordered_set<string> wordSet(wordDict.begin(), wordDict.end());
	wordBreak_dfs2(s, m, wordSet, 0);
	return m[0];
}

//141 环形链表
/*
我们定义两个指针，一快一满。慢指针每次只移动一步，而快指针每次移动两步.
初始时，慢指针在位置 head，而快指针在位置 head.next。
这样一来，如果在移动的过程中，快指针反过来追上慢指针，就说明该链表为环形链表。
否则快指针将到达链表尾部，该链表不为环形链表。
*/
bool hasCycle(Lnode *head) {
	if (head == NULL || head->next == NULL)return false;
	Lnode* fast = head->next;
	Lnode* slow = head;
	while (fast != slow) {
		if (fast == NULL || fast->next == NULL)return false;
		fast = fast->next->next;
		slow = slow->next;
	}
	return true;
}

//148 排序链表
Lnode* cut(Lnode* head, int n) {
	//将链表 l 切掉前 n 个节点，并返回后半部分的链表头。
	auto p = head;
	n--;
	while (n&&p) {
		n--;
		p = p->next;
	}
	if (!p)return nullptr;
	//从p处断开链表，返回断开后面的链表的第一个节点
	auto next = p->next;
	p->next = nullptr;
	return next;
}
Lnode* merge(Lnode* l1, Lnode* l2) {
	//返回合并后链表的头节点
	Lnode dunmmyhead(0);
	auto p = &dunmmyhead;
	while (l1&&l2) {
		if (l1->val < l2->val) {
			p->next = l1;
			p = l1;
			l1 = l1->next;
		}
		else {
			p->next = l2;
			p = l2;
			l2 = l2->next;
		}
	}
	p->next = l1 ? l1 : l2;
	return dunmmyhead.next;
}
Lnode* sortList(Lnode* head) {
	Lnode dunmmyhead(0);
	dunmmyhead.next = head;
	auto p = head;
	int len = 0;
	while (p) {
		len++;
		p = p->next;
	}

	for (int size = 1; size < len; size <<= 1) {
		auto cur = dunmmyhead.next;
		auto tail = &dunmmyhead;

		while (cur) {
			//left->@->@->@->@->@->@->null
			auto left = cur;
			// left->@->@->null   right->@->@->@->@->null
			auto right = cut(left, size);
			// left->@->@->null   right->@->@->null   current->@->@->null
			cur = cut(right, size);
			// dummy.next -> @->@->@->@->null，最后一个节点是 tail，始终记录
			//                        ^
			//                        tail
			tail->next = merge(left, right);
			// 保持 tail 为尾部
			while (tail->next)tail = tail->next;
		}
	}
	return dunmmyhead.next;
}

//149 直线上最多的点数
/*
(x1,y1) (x2,y2) b = (x2y1-x1y2)/(x2-x1) k = (y2-y1)/(x2-x1)
unordered_map<vector<int>, int> map1; 错误
	我们知道c++中有unordered_map和unordered_set这两个数据结构，
	其内部实现是哈希表，这就要求作为键的类型必须是可哈希的，一般来说都是基本类型
	所以pair和vector一类的不可以
map<vector<int>, int> map2;  正确
	map和set内部的实现是树（红黑树，更高级的二叉搜索树），
	既然运用到二叉搜索树就必须得有比较两个元素大小的方法，
	所以pair<int,int>和vector<int>，可以直接作为键进行使用
unordered_map<int, vector<int>> map;  哈希表的值可以是数组等其他复杂类型
	可以用一个哈希表记录多种内容，而不必使用多个哈希表
*/
int maxPoints(vector<vector<int>>& points) {
	if (points.size() == 1)return 1;
	map<vector<double>, unordered_set<int>> m;
	int ans = 0;
	for (int i = 0; i < points.size() - 1; i++) {
		for (int j = i + 1; j < points.size(); j++) {
			vector<double> t;
			if ((points[j][0] - points[i][0]) != 0 && (points[j][1] - points[i][1]) != 0) {
				double k = double((points[j][1] - points[i][1])) / double((points[j][0] - points[i][0]));
				double b = double(points[j][0] * points[i][1] - points[i][0] * points[j][1]) / double(points[j][0] - points[i][0]);
				t = { k,b };
			}
			else if ((points[j][0] - points[i][0]) == 0) {
				t = { DBL_MIN,double(points[i][0]) };
			}
			else {
				t = { DBL_MAX,double(points[i][1]) };
			}
			if (m.count(t)) {
				m[t].insert(j);
				m[t].insert(i);
			}
			else m.insert(pair<vector<double>, unordered_set<int>>(t, { i,j }));
		}
	}
	for (auto pointk : m) {
		ans = fmax(pointk.second.size(), ans);
	}
	return ans;
}

//155 最小栈
/*
我们只需要设计一个数据结构，使得每个元素 a 与其相应的最小值 m 时刻保持一一对应。
因此我们可以使用一个辅助栈，与元素栈同步插入与删除，用于存储与每个元素对应的最小值。
*/
class MinStack {
public:
	stack<int> s;
	stack<int> h;
	MinStack() {
	}

	void push(int val) {
		s.push(val);
		if (h.empty())h.push(val);
		else h.push(min(val, h.top()));
	}

	void pop() {
		s.pop();
		h.pop();
	}

	int top() {
		return s.top();
	}

	int getMin() {
		return h.top();
	}
};


//162 寻找峰值
/*
起点是负无穷，开始一定是上坡，目标是寻找序列中第一个下降点，序列从左到右是从“不满足”状态到“满足”状态的。
如果nums[mid] < nums[mid+1]，说明仍然不满足，不必包含mid，继续向右找，即l = mid +1；
如果nums[mid] > nums[mid+1]，说明此时这个mid位置满足了，但不一定是第一个满足的，所以要把mid包含在内，向左找，即r = mid；
退出条件是l == r，也就是框出了唯一的一个位置，此时退出，返回l即可。这是一个很经典的二分框架～
*/
int findPeakElement(vector<int>& nums) {
	int l = 0, r = nums.size() - 1;
	while (l < r) {
		int mid = l + (r - l) / 2;
		if (nums[mid] > nums[mid + 1])r = mid;
		else if (nums[mid] < nums[mid + 1])l = mid + 1;
	}
	return l;
}


//166 分数到小数 长除法
string fractionToDecimal(int numerator, int denominator) {
	string ans;
	if ((numerator < 0 && denominator>0) || (numerator > 0 && denominator < 0))ans.push_back('-');
	long m_numerator = abs(numerator);
	long m_denominator = abs(denominator);
	ans = ans + (to_string(m_numerator / m_denominator));
	if (m_numerator%m_denominator == 0)return ans;

	ans.push_back('.');
	//正数部分长度
	int parton1 = ans.size();
	//每一个余数 对应的第几次的余数
	unordered_map<long, int> m;
	long render = m_numerator % m_denominator;
	int i = 1;
	while (render != 0 && !m.count(render)) {
		m[render] = i;
		m_numerator = render * 10;
		render = m_numerator % m_denominator;
		ans.push_back(m_numerator / m_denominator + '0');
		i++;
	}

	if (render == 0)return ans;
	//出现重复余数的位置
	int index = m[render];
	string res = "";
	res = ans.substr(0, parton1 + index - 1) + "(" + ans.substr(parton1 + index - 1) + ")";
	return  res;
}

//169 多数元素
//出现次数超过N/2的数字 水王数   摩尔投票思想
//时间复杂度 o(n) 空间复杂度 o(1)
int t31(vector<int>& nums) {
	//核心 每次删除俩个不同的数字
	//最后如果剩下一个数字，再遍历一遍 统计和剩下的相等的数字的个数
	int tar = 0;
	int count = 0;
	for (int i = 0; i < nums.size(); i++) {
		//count = 0 认为没有目标
		//需要 更新目标和count
		if (count = 0) {
			tar = nums[i];
			count = 1;
		}
		else {
			//有目标时 
			//如果当前数字等于目标 不删 且 count++
			if (nums[i] == tar)count++;
			//如果当前数字不等于目标 删除 且 count--
			else count--;
		}
	}
	//如果count = 0 不存在水王数
	if (count == 0)return -1;
	//否则 再数组中统计和目标值相等的数字的个数
	count = 0;
	for (int i = 0; i < nums.size(); i++) {
		if (nums[i] == tar)count++;
	}
	if (count > nums.size() / 2)return tar;
	else return -1;
}

//查找替换 构造双射  双射要求 一一对应  
class Solution890 {
	//map 保证了key的唯一 可能出现多对一情况，
	//所以 需要match两次 交换key 和 value 才可以保证一一对应
	bool match(string &word, string &pattern) {
		unordered_map<char, char> mp;
		for (int i = 0; i < word.length(); ++i) {
			char x = word[i], y = pattern[i];
			if (!mp.count(x)) {
				mp[x] = y;
			}
			else if (mp[x] != y) { // word 中的同一字母必须映射到 pattern 中的同一字母上
				return false;
			}
		}
		return true;
	}

public:
	vector<string> findAndReplacePattern(vector<string> &words, string &pattern) {
		vector<string> ans;
		for (auto &word : words) {
			if (match(word, pattern) && match(pattern, word)) {
				ans.emplace_back(word);
			}
		}
		return ans;
	}
};


//172  阶乘后的0的个数
int t43(int n) {
	/*
	实际需要计算从1到n中的数字 因子2和因子5的最小数目
	又由于 因子2的个数一定大于因子5的个数
	实际 是计算1到n中的所有数字 每个数子的因子5的个数的和
	从1到n 每5个数有一个因数5
	从1到n 每25个数多一个因数5
	从1到n 每125个数多一个因数5
	......
	所以 公式 n/5 + n/25 + n/125+...
	*/
	int res = 0;
	while (n) {
		res = res + n / 5;
		n = n / 5;
	}
	return res;
}

//最大数
class Solution179 {
public:
	static bool mycmp(int num1, int num2) {
		string a = to_string(num1) + to_string(num2);
		string b = to_string(num2) + to_string(num1);
		return a > b;
	}
	string largestNumber(vector<int>& nums) {
		//如果全为0 返回0
		int cout0 = 0;
		string ans;
		sort(nums.begin(), nums.end(), mycmp);
		for (int i : nums) {
			if (i == 0)cout0++;
			ans += (to_string(i));
		}
		if (cout0 == nums.size())return "0";
		return ans;
	}
};

//189 轮转数组
class Solution {
public:
	void revers(vector<int>& nums, int st, int ed) {
		while (st <= ed) {
			swap(nums[st], nums[ed]);
			st++;
			ed--;
		}
	}
	void rotate(vector<int>& nums, int k) {
		k = k % nums.size();
		revers(nums, 0, nums.size() - 1);
		revers(nums, 0, k - 1);
		revers(nums, k, nums.size() - 1);
	}
};

//颠倒二进制位190  uint32_t  00000010100101000001111010011100
uint32_t t44(uint32_t t) {
	t = t >> 16 | t << 16;//高低16位互换

	t = ((t & 0xff00ff00) >> 8) | ((t & 0x00ff00ff) << 8);//每16位一组 一组中的高低8位互换
	t = ((t & 0xf0f0f0f0) >> 4) | ((t & 0x0f0f0f0f) << 4);//每8位一组  一组中的高低4位互换
	t = ((t & 0xcccccccc) >> 2) | ((t & 0x33333333) << 2);//每4位一组  一组中的高低2位互换
	t = ((t & 0xaaaaaaaa) >> 1) | ((t & 0x55555555) << 1);//每2位一组  一组中的高低1位互换

	return t;
}
uint32_t reverseBits(uint32_t n) {
	uint32_t res = 0;
	for (int i = 0; i <= 31; i++) {
		uint32_t tmp = 0;
		//从低到高 判断某一位是否为1
		if ((n&(1 << i)) > 0)tmp = 1;
		res = res + (tmp << (31 - i));
	}
	return res;
}


//打家劫舍2  0到n构成环
int t47(vector<int>&nums) {
	if (nums.size() == 1)return nums[0];
	if (nums.size() == 2)return max(nums[1], nums[0]);
	vector<int> dp1(nums.size() - 1);
	vector<int> dp2(nums.size());
	//从0到n-1 不要n
	dp1[0] = nums[0];
	dp1[1] = max(nums[0], nums[1]);
	for (int i = 2; i < nums.size() - 1; i++) {
		//只要当前家
		int p1 = nums[i];
		//不要当前家
		int p2 = dp1[i - 1];
		//要当前家
		int p3 = dp1[i - 2] + nums[i];
		dp1[i] = max(p1, max(p2, p3));
	}
	//从1到n  不要0
	dp2[1] = nums[1];
	dp2[2] = max(nums[1], nums[2]);
	for (int i = 3; i < nums.size(); i++) {
		//只要当前家
		int p1 = nums[i];
		//不要当前家
		int p2 = dp2[i - 1];
		//要当前家
		int p3 = dp2[i - 2] + nums[i];
		dp2[i] = max(p1, max(p2, p3));
	}
	return max(dp1.back(), dp2.back());
}

//岛屿数量
class Solution200 {
public:
	void numIslands_dfs(vector<vector<char>>& grid, int i, int j, int m, int n) {
		grid[i][j] = '0';
		if (i + 1 < m&&grid[i + 1][j] == '1')numIslands_dfs(grid, i + 1, j, m, n);
		if (j + 1 < n&&grid[i][j + 1] == '1')numIslands_dfs(grid, i, j + 1, m, n);
		if (i - 1 >= 0 && grid[i - 1][j] == '1')numIslands_dfs(grid, i - 1, j, m, n);
		if (j - 1 >= 0 && grid[i][j - 1] == '1')numIslands_dfs(grid, i, j - 1, m, n);
	}
	int numIslands(vector<vector<char>>& grid) {
		int ans = 0;
		int m = grid.size(), n = grid[0].size();
		for (int r = 0; r < m; r++) {
			for (int c = 0; c < n; c++) {
				if (grid[r][c] == '1') {
					ans++;
					numIslands_dfs(grid, r, c, m, n);
				}
			}
		}
		return ans;
	}
};

//202 快乐数
bool isHappy(int n) {
	int res = 0;
	//set记录出现过的数，如果某个数重复出现 则进入循环 ，直接返回false
	unordered_set<int> Set;
	Set.insert(n);
	while (1) {
		while (n) {
			res = res + pow(n % 10, 2);
			n = n / 10;
		}
		if (res == 1)return true;
		n = res;
		if (Set.insert(n).second == false)return false;
		res = 0;
	}
	return false;
}

//204 计数质数 小于n的质数
int countPrimes(int n) {
	int ans = 0;
	vector<bool> isprimes(n, true);
	for (int i = 2; i < n; i++) {
		if (isprimes[i]) {
			ans++;
			//防止溢出
			if ((long long)i*i >= n)continue;
			for (int j = i * i; j < n; j = j + i)isprimes[j] = false;
		}
	}
	return ans;
}

//206 反转链表
Lnode* transLink(Lnode* head) {
	//链表逆序
	Lnode* next = nullptr;
	Lnode* pre = nullptr;
	while (!head)
	{
		next = head->next;//记录当前结点的下一个结点
		head->next = pre;//当前结点的next指向pre
		pre = head;//pre向前移动指向当前结点
		head = next;//当前结点也向前移动指向被记录下来的结点
	}
	return pre;
}

//207 课程表 拓扑排序
bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
	//key 是value中元素所依赖的课程
	unordered_map<int, vector<int>> edges;
	vector<int> in(numCourses, 0);
	for (int i = 0; i < prerequisites.size(); i++) {
		edges[prerequisites[i][1]].push_back(prerequisites[i][0]);
		in[prerequisites[i][0]]++;
	}
	queue<int> q;
	int learn = 0;
	for (int i = 0; i < numCourses; i++) {
		if (in[i] == 0)q.push(i);
	}
	while (!q.empty()) {
		int cur = q.front();
		q.pop();
		learn++;
		for (int next : edges[cur]) {
			in[next]--;
			if (in[next] == 0)q.push(next);
		}
	}
	return learn == numCourses;
}

//208 字典树
class Trie {
private:
	vector<Trie*> next;
	bool isEnd;
public:

	Trie() : next(26), isEnd(false) {}

	void insert(string word) {
		Trie* tmp = this;
		int i = 0;
		while (i < word.size()) {
			if (tmp->next[word[i] - 'a'] == nullptr) {
				tmp->next[word[i] - 'a'] = new Trie;
			}
			tmp = tmp->next[word[i] - 'a'];
			i++;
		}
		tmp->isEnd = true;
	}

	bool search(string word) {
		Trie* tmp = this;
		int i = 0;
		while (i < word.size()) {
			if (tmp->next[word[i] - 'a'] == nullptr)return false;
			tmp = tmp->next[word[i] - 'a'];
			i++;
		}
		return tmp->isEnd;
	}

	bool startsWith(string prefix) {
		Trie* tmp = this;
		int i = 0;
		while (i < prefix.size()) {
			if (tmp->next[prefix[i] - 'a'] == nullptr)return false;
			tmp = tmp->next[prefix[i] - 'a'];
			i++;
		}
		return true;
	}
};
//210 课程表2
vector<int> findOrder(int k, vector<vector<int>>& prerequisites) {
	vector<vector<int>> g(k);
	vector<int> inDegree(k, 0);
	vector<int> ans;
	for (int i = 0; i < prerequisites.size(); i++) {
		inDegree[prerequisites[i][0]]++;
		g[prerequisites[i][1]].push_back(prerequisites[i][0]);
	}
	//入度为0的先入队列
	queue<int> q;
	for (int i = 0; i < k; i++) {
		if (inDegree[i] == 0)q.push(i);
	}
	while (!q.empty()) {
		ans.push_back(q.front());
		int cur = q.front();
		q.pop();
		for (int next : g[cur]) {
			inDegree[next]--;
			if (inDegree[next] == 0)q.push(next);
		}
	}
	if (ans.size() != k)return {};
	return ans;
}

//天际线问题 扫描线+优先队列  困难
/*
// 如果将所有的建筑的边界作为一条线，那么所有的答案都在这些线上
// 考虑任意一条线，那么这条线和所有相交的建筑（这里排除掉刚好和建筑右边界相交），取一个最高的
// 高度，然后判断这个高度是否和ans末尾最后一个元素的高度相等，不相等就加入进去
// 在这里为了快速得到最高的高度，使用一个堆来进行记录
*/
class Solution218 {
public:
	//定义根据高度降序排列
	struct cmp {
		bool operator()(pair<int, int>&a, pair<int, int>&b) {
			return a.second < b.second;
		}
	};
	vector<vector<int>> getSkyline(vector<vector<int>>& buildings) {
		vector<int> bands;//所有建筑的左边界
		vector<vector<int>>ans;
		//存储右边界-高度
		priority_queue<pair<int, int>, vector<pair<int, int>>, cmp> p;
		for (int i = 0; i < buildings.size(); i++) {
			bands.push_back(buildings[i][0]);
			bands.push_back(buildings[i][1]);
		}
		sort(bands.begin(), bands.end());

		int index = 0;//指向bulidings
		for (int i = 0; i < bands.size(); i++) {
			// 对于一个建筑，如果其左边界在当前判断的边界线左边或重叠，那么向堆加入右边界-高度值对
			while (index < buildings.size() && buildings[index][0] <= bands[i]) {
				p.push({ buildings[index][1],buildings[index][2] });
				index++;
			}
			// 对于那些加入了堆中的建筑，从堆的顶部移出建筑右边界在边界线左边或重叠的边界-高度值对
			while (!p.empty() && p.top().first <= bands[i]) {
				p.pop();
			}
			// 如果此时的堆为空，证明边界线没有穿过任何建筑，来到了建筑的分割位置，天际线为0
			int maxhight = p.empty() ? 0 : p.top().second;
			// 按照这种算法，每一条边界线都会产生一个天际线高度，如果这个高度和ans末尾元素的高度一致，
			//那么就说明两条边界线穿过了同一个建筑，并且相邻，那么按照规则只取最左端
			if (ans.size() == 0 || maxhight != ans.back()[1])ans.push_back({ bands[i],maxhight });
		}
		return ans;
	}
};

//计算器  数学表达式字符串的结果计算（没有括号） 
class Solution227 {
public:

	int calculate(string s) {
		int ans = 0;
		vector<int> num;
		//保存一个完整数字的前面的符号
		char persing = '+';
		int tmp = 0;
		for (int i = 0; i < s.size(); i++) {
			if (isdigit(s[i])) {
				tmp = tmp * 10 + (s[i] - '0');
			}
			//得到了一个完整数字
			if (!isdigit(s[i]) && s[i] != ' ' || i == s.size() - 1) {
				//判断这个数字前面的符号
				if (persing == '+') {
					num.push_back(tmp);
				}
				else if (persing == '-') {
					num.push_back(-tmp);
				}
				else if (persing == '*') {
					num.back() = num.back()*tmp;
				}
				else if (persing == '/') {
					num.back() = num.back() / tmp;
				}
				//更新符号和数字

				persing = s[i];
				tmp = 0;
			}
		}
		for (int i : num)ans += i;
		return ans;
	}
};
//数学表达式字符串的结果计算（有括号）  递归
vector<int> t18(string& str, int index) {
	stack<string> s;
	int t_num = 0;
	string t_str;
	while (index < str.size() && str[index] != ')') {
		if (str[index] >= '0'&&str[index] <= '9') {
			t_num = t_num * 10 + str[index] - '0';
			index++;
		}
		else if (str[index] != '(') {
			//遇到运算符，就把前面的一个数值和当前运算符入栈
			//如果栈顶是 *  /  则需要把栈顶的符号和数字都拿出来，做完运算后再放入
			if (!s.empty() && s.top() == "*") {
				s.pop();
				string tmp = s.top();
				s.pop();
				s.push(to_string(stoi(tmp)*t_num));

			}
			else if (!s.empty() && s.top() == "/") {
				s.pop();
				string tmp = s.top();
				s.pop();
				s.push(to_string(stoi(tmp) / t_num));
			}
			else {
				t_str.push_back(str[index]);
				s.push(to_string(t_num));
				s.push(t_str);
				t_str = "";
				t_num = 0;
			}
			index++;
		}
		else {
			//遇到 ( 开始递归 
			vector<int> p = t18(str, index + 1);
			t_num = p[0];
			index = p[1] + 1;
		}
	}
	//计算目前栈内的所有数据表达式的结果，并返回
	//先把最后一个元素放进去，放元素如果栈顶是 *  /  
	//则先把栈顶的  运算符 和 当前数字 还有栈顶的符号下的数字计算完后再放进去
	if (!s.empty() && s.top() == "*") {
		s.pop();
		string tmp = s.top();
		s.pop();
		s.push(to_string(stoi(tmp)*t_num));

	}
	else if (!s.empty() && s.top() == "/") {
		s.pop();
		string tmp = s.top();
		s.pop();
		s.push(to_string(stoi(tmp) / t_num));
	}
	else {
		s.push(to_string(t_num));
		t_str = "";
		t_num = 0;
	}
	//此时栈中只有 + - 运算，全部拿出来，从左到右计算
	vector<string> str_list;
	while (!s.empty()) {
		str_list.push_back(s.top());
		s.pop();
	}
	int tmp = 0;
	reverse(str_list.begin(), str_list.end());
	for (int i = 0; i < str_list.size() - 1; i++) {
		if (str_list[i] == "-") {
			tmp = stoi(str_list[i - 1]) - stoi(str_list[i + 1]);
			str_list[i + 1] = to_string(tmp);
		}
		else if (str_list[i] == "+") {
			tmp = stoi(str_list[i - 1]) + stoi(str_list[i + 1]);
			str_list[i + 1] = to_string(tmp);
		}
	}
	return { stoi(str_list.back()),index };
}

//二叉树的第k小元素 230
// 注意  k是引用，采用中序遍历 二叉搜索树的性质
int kthSmallestdfs(Tnode* root, int& k) {
	if (root == nullptr) return -1;
	int left = kthSmallestdfs(root->left, k);
	k--;
	if (k == 0) {
		return root->val;
	}
	int right = kthSmallestdfs(root->right, k);
	if (left == -1)return right;
	else return left;
}

//234 判断链表是否回文
bool isPalindrome(Lnode* head) {
	Lnode* slow = head;
	Lnode* pre = nullptr;
	Lnode* fast = head;
	//双指针找中点
	while (fast) {
		slow = slow->next;
		fast = fast->next ? fast->next->next : fast->next;
	}
	//后半段逆序
	while (slow) {
		Lnode* tmp = slow->next;
		slow->next = pre;
		pre = slow;
		slow = tmp;
	}

	while (head&&pre) {
		if (head->val == pre->val) {
			head = head->next;
			pre = pre->next;
		}
		else return false;
	}
	return true;
}

//236 二叉树的最近公共祖先
Tnode* lowestCommonAncestor(Tnode* root, Tnode* p, Tnode* q) {
	if (root == NULL)return NULL;
	if (root == p)return p;
	if (root == q)return q;

	Tnode* left = lowestCommonAncestor(root->left, p, q);
	Tnode* right = lowestCommonAncestor(root->right, p, q);
	if (left&&right)return root;//p q 分别在两边
	else if (left)return left;// p q 都在左边
	else if (right)return right;// p q 都在右边
	else return NULL;
}

//删除链表中的节点
void deleteNode237(Lnode* node) {
	node->val = node->next->val;
	node->next = node->next->next;
}

//238 除自身外的数组乘积
vector<int> productExceptSelf(vector<int>& nums) {
	int size = nums.size();
	vector<int> ans(size);
	vector<int> st(size);
	vector<int> ed(size);
	st[0] = nums[0];
	ed.back() = nums.back();
	for (int i = 1; i < size; i++) {
		st[i] = st[i - 1] * nums[i];
	}
	for (int i = size - 2; i >= 0; i--) {
		ed[i] = ed[i + 1] * nums[i];
	}
	ans[0] = ed[1];
	ans.back() = st[size - 2];
	for (int i = 1; i < size - 1; i++) {
		ans[i] = ed[i + 1] * st[i - 1];//1 2 6 24    24 24 12 4
	}
	return ans;
}


//287 寻找重复的数
/*
我们对nums 数组建图，每个位置 i 连一条 i→nums[i] 的边。由于存在的重复的数字 target，
因此 target 这个位置一定有起码两条指向它的边，因此整张图一定存在环，且我们要找到的 target 就是这个环的入口，
那么整个问题就等价于 142. 环形链表 II。
*/
int findDuplicate(vector<int>& nums) {
	int slow = 0;
	int fast = 0;
	slow = nums[slow];
	fast = nums[nums[fast]];
	//有环的情况下一定会相遇
	while (slow != fast) {
		slow = nums[slow];
		fast = nums[nums[fast]];
	}
	slow = 0;
	//找入环口
	while (slow != fast) {
		slow = nums[slow];
		fast = nums[fast];
	}
	return slow;
}

//239滑动窗口的最大值
 //双端队列  队列中存下标
 //时刻保持队头是区间最大元素下标
//每个元素下标从队尾插入
//要保持插入后的顺序是单调减（从队头到队尾递减），如果不满足就从队尾弹出，弹出到满足之后在插入
//插入后判断队头元素下标是否还在窗口中，如果不在就弹出
//如果在它就是窗口中的最大元素的下标
vector<int> maxSlidingWindow(vector<int>& nums, int k) {
	deque<int> q;
	vector<int> ans;
	q.push_back(0);//先插入一个元素
	//窗口达到最大
	for (int i = 1; i < k; i++) {
		while (!q.empty() && nums[q.back()] <= nums[i]) {
			q.pop_back();
		}
		q.push_back(i);
	}
	//收集第一个答案
	ans.push_back(q.front());
	//窗口开始向右移动
	for (int i = k; i < nums.size(); i++) {
		while (!q.empty() && nums[q.back()] <= nums[k]) {
			q.pop_back();
		}
		q.push_back(i);
		//判断队头元素是否还在窗口中 窗口的左边界 i-k+1
		if (q.front() < i - k + 1)q.pop_front();
		ans.push_back(q.front());
	}
	return ans;
}

//240搜索二维矩阵
bool searchMatrix(vector<vector<int>>& matrix, int target) {
	int m = matrix.size();
	int n = matrix[0].size();
	int i = 0, j = n - 1;
	while (i < m&&j >= 0) {
		if (matrix[i][j] == target)return true;
		else if (matrix[i][j] < target)i++;
		else if (matrix[i][j] > target)j--;
	}
	return false;
}

//268 丢失的数字
int missingNumber(vector<int>& nums) {
	int k = nums.size();
	for (int i = 0; i < nums.size();) {
		if (nums[i] == i || nums[i] == k)i++;
		else swap(nums[i], nums[nums[i]]);
	}
	for (int i = 0; i < nums.size(); i++) {
		if (nums[i] != i)return i;
	}
	return k;
}

//279完全平方数  完全背包问题
int t49(int n) {
	vector<int> dp(n + 1, INT_MAX);
	dp[0] = 0;
	for (int i = 1; i <= n; i++) {
		for (int j = 1; j*j <= i; j++) {
			//动态转移方程 好好理解！！！
			dp[i] = min(dp[i - j * j] + 1, dp[i]);
		}
	}
	return dp.back();
}


//283 移动零
/*
使用双指针，左指针指向当前已经处理好的序列的尾部，右指针指向待处理序列的头部。

右指针不断向右移动，每次右指针指向非零数，则将左右指针对应的数交换，同时左指针右移。

*/
void moveZeroes(vector<int>& nums) {
	int i = 0, j = 0;
	for (i; i < nums.size(); i++) {
		if (nums[i] != 0) {
			swap(nums[i], nums[j]);
			j++;
		}
	}
	return;
}

//找中位数
/*
大堆和小堆，每一时刻两个堆的大小满足 大堆+1=小堆，或者大堆=小堆
如果小堆为空或者新数小于等于小堆的对顶 放入小堆
否则 放入大堆
放入后 如果不满足两个堆的大小要求则需要调整（大堆/小堆的堆顶元素移动至小堆/大堆） 

某一时刻的中位数是
如果 大堆+1=小堆 则返回小堆堆顶元素
如果 大堆=小堆 则返回两个堆顶元素的和的一半
*/

class MedianFinder295 {
public:
	priority_queue<int, vector<int>, less<int>> qmin;
	priority_queue<int, vector<int>, greater<int>> qmax;
	MedianFinder295() {}

	void addNum(int num) {
		if (qmin.empty() || num <= qmin.top()) {
			qmin.push(num);
			if (qmax.size() + 1 < qmin.size()) {
				qmax.push(qmin.top());
				qmin.pop();
			}
		}
		else {
			qmax.push(num);
			if (qmax.size() > qmin.size()) {
				qmin.push(qmax.top());
				qmax.pop();
			}
		}
	}

	double findMedian() {
		if (qmin.size() > qmax.size())return qmin.top();
		else return (qmin.top() + qmax.top()) / 2.0;
	}
};

//二叉树的序列化和反序列化
class Codec297 {
public:

	// Encodes a tree to a single string.
	string serialize(Tnode* root) {
		string res;
		myserialize(root, res);
		return res;
	}
	void myserialize(Tnode* root, string& str) {
		if (root == NULL)str += "none,";
		else {
			str += to_string(root->val) + ",";
			myserialize(root->left, str);
			myserialize(root->right, str);
		}
	}

	// Decodes your encoded data to tree.
	Tnode* mydeserialize(list<string>& liststr) {
		if (liststr.front() == "none") {
			liststr.erase(liststr.begin());
			return NULL;
		}
		Tnode* root = new Tnode(stoi(liststr.front()));
		liststr.erase(liststr.begin());
		root->left = mydeserialize(liststr);
		root->right = mydeserialize(liststr);
		return root;

	}
	Tnode* deserialize(string data) {
		//把字符串拆分成字符串链表
		list<string> liststr;
		string str;
		for (char& c : data) {
			if (c == ',') {
				liststr.push_back(str);
				str.clear();
			}
			else {
				str.push_back(c);
			}
		}
		if (!str.empty()) {
			liststr.push_back(str);
			str.clear();
		}
		return mydeserialize(liststr);
	}
};


int t2(vector<int>& nums) {
	//最长递增子序列
	//dp[i] 以nums[i]结尾的最长递增子序列的长度
	//end[i] 目前所有长度为 i+1 的递增子序列中的最小序列结尾
	//再有效区中二分查找刚好大于nums[i]的值，并用nums[i]替换
	vector<int> dp(nums.size());
	vector<int> end(nums.size());
	dp[0] = 1, end[0] = nums[0];
	//l，r 维持end数组的有效区
	int l = 0, r = 0;
	vector<int>::iterator it;
	for (int i = 1; i < nums.size(); i++) {
		//找第一个大于等于的位置,查找区间 [ )
		it = lower_bound(end.begin() + l, end.begin() + r + 1, nums[i]);
		//如果找到 nums[i] 替换找到的数值
		if (it != end.begin() + r + 1) {
			*it = nums[i];
			dp[i] = end.begin() - it;
		}
		//如果没找到 有效区扩增一个新的元素 nums[i]
		else {
			r++;
			end[r] = nums[i];
			dp[i] = r - l + 1;
		}
	}
	return r - l + 1;
}
int lengthOfLIS(vector<int>& nums) {
	//dp[i] 为考虑前 i 个元素，以第 i 个数字结尾的最长上升子序列的长度
	vector<int> dp(nums.size(), 1);
	for (int i = 0; i < nums.size(); i++) {
		for (int j = 0; j < i; j++) {
			if (nums[i] > nums[j]) {
				dp[i] = max(dp[i], dp[j] + 1);
			}
		}
	}
	return *max_element(dp.begin(), dp.end());
}

//324 摆动序列
void wiggleSort(vector<int>& nums) {
	if (nums.size() == 1)return;
	vector<int> tmp = nums;
	sort(tmp.begin(), tmp.end());
	//如果数组长度为奇数，则前半部分要比后半部分多一个
	//为了能够正确排序，两个部分要逆序后在穿插
	int l = (tmp.size() - 1) / 2, r = (tmp.size()) - 1;
	int i = 0;
	while (i < nums.size()) {
		nums[i++] = tmp[l--];
		nums[i++] = tmp[r--];
	}
}
//为了降低时间复杂度，可以进行部分排序，实际上不需要对数组完全排序
//找到中位数，在把数组按照中位数做一次partiton即可完成部分排序
//找中位数实际上是找第k大的数的应用
void wiggleSort2(vector<int>& nums) {
	if (nums.size() == 1)return;
	auto midptr = nums.begin() + nums.size() / 2;
	//nth_element部分排序函数，传入三个迭代器参数
	nth_element(nums.begin(), midptr, nums.end());
	int mid = *midptr;
	//partition
	int l = 0, r = nums.size() - 1, k = 0;
	while (k < r) {
		if (nums[k] > mid) {
			swap(nums[k], nums[r]);
			r--;
		}
		else if (nums[k] < mid) {
			swap(nums[k], nums[l]);
			l++;
			k++;
		}
		else k++;
	}
	//如果数组长度为奇数，则前半部分要比后半部分多一个
	//为了能够正确排序，两个部分要逆序后在穿插
	vector<int> tmp = nums;
	l = (tmp.size() - 1) / 2, r = (tmp.size()) - 1;
	int i = 0;
	while (i < nums.size()) {
		nums[i++] = tmp[l--];
		nums[i++] = tmp[r--];
	}
}

//328 奇数偶数链表 根据索引分割链表
Lnode* oddEvenList(Lnode* head) {
	if (head == nullptr)return nullptr;
	Lnode* p1 = head;
	Lnode* p1h = p1;
	Lnode* p2h = nullptr;
	if (head->next != nullptr) {
		Lnode* p2 = head->next;
		p2h = p2;
		while (p2&&p1) {
			p1->next = p2->next;
			if (p1->next)p1 = p1->next;
			p2->next = p1->next;
			p2 = p2->next;
		}
	}
	p1->next = p2h;
	return p1h;
}

//329 矩阵中的最长递增路径   
class Solution329 {
public:
	int dir[4][2] = { {0,1},{0,-1},{1,0},{-1,0} };
	int r, c;
	int h_dfs(vector<vector<int>>& matrix, int i, int j, vector<vector<int>>& dp) {
		if (dp[i][j] != -1)return dp[i][j];

		dp[i][j]++;
		for (int k = 0; k < 4; k++) {
			int curr = i + dir[k][0];
			int curc = j + dir[k][1];
			//递归的条件 matrix[curr][curc]>matrix[i][j] 决定了不会出现重复访问的点 
			//因为不会重复访问一个点，所以不用used，不用used也就用不到回溯
			if (curr >= 0 && curr < r&&curc >= 0 && curc<c&&matrix[curr][curc]>matrix[i][j])
				//这个点的最大路径 = 向4个方向上寻找到的最大路径 + 1 
				dp[i][j] = max(dp[i][j], h_dfs(matrix, curr, curc, dp) + 1);
		}
		return dp[i][j];
	}
	int longestIncreasingPath(vector<vector<int>>& matrix) {
		vector<vector<int>>dp(matrix.size(), vector<int>(matrix[0].size(), -1));
		r = matrix.size();
		c = matrix[0].size();
		int tmax = 0;
		for (int i = 0; i < r; i++) {
			for (int j = 0; j < c; j++) {
				tmax = max(tmax, h_dfs(matrix, i, j, dp));
			}
		}
		return tmax + 1;
	}
};

//334 递增3元组
/*
first 始终记录最小元素，second 为某个子序列里第二大的数。

接下来不断更新 first，同时保持 second 尽可能的小。

如果下一个元素比 second 大，说明找到了三元组。
*/
bool increasingTriplet(vector<int>& nums) {
	int first = INT_MAX;
	int second = INT_MAX;
	for (int i = 0; i < nums.size(); i++) {
		if (nums[i] <= first) {
			first = nums[i];
		}
		else if (nums[i] <= second) {
			second = nums[i];
		}
		else return true;
	}
	return false;
}
bool increasingTripletdp(vector<int>& nums) {

	//记录每个元素前面的最小值，和后面的最大值
	//如果 前面的最小值、当前元素、后面的最大值满足递增 即返回true
	vector<int> premin(nums.size(), INT_MAX);
	vector<int> nextmax(nums.size(), INT_MIN);

	for (int i = 1; i < nums.size(); i++) {
		premin[i] = min(nums[i - 1], premin[i - 1]);
	}
	for (int i = nums.size() - 2; i >= 0; i--) {
		nextmax[i] = max(nums[i + 1], nextmax[i + 1]);
	}
	for (int i = 1; i < nums.size() - 1; i++) {
		if (premin[i] < nums[i] && nums[i] < nextmax[i])return true;
	}
	return false;
}

class NestedInteger {
public:
	bool isInteger() { return true; }
	int getInteger() { return 0; }
	vector<NestedInteger> getList(){}

};
/*
嵌套的整型列表是一个树形结构，树上的叶子节点对应一个整数，非叶节点对应一个列表。
在这棵树上深度优先搜索的顺序就是迭代器遍历的顺序。
我们可以先遍历整个嵌套列表，将所有整数存入一个数组，然后遍历该数组从而实现next 和hasNext 方法。
*/
class NestedIterator {
public:
	vector<int> val;
	vector<int>::iterator it;
	void dfs(const vector<NestedInteger> &nestedList) {
		for (auto nest : nestedList) {
			if (nest.isInteger())val.push_back(nest.getInteger());
			else dfs(nest.getList());
		}
	}
	NestedIterator(vector<NestedInteger> &nestedList) {
		dfs(nestedList);
		it = val.begin();
	}

	int next() {
		return *it++;
	}

	bool hasNext() {
		return it != val.end();
	}
};

//前k个高频元素
//如果不用堆则需要对所有元素排序 时间复杂度logN，使用堆将减小到logk
class Solution347 {
public:
	static bool cmp(vector<int>&a, vector<int>&b) {
		return a[1] > b[1];
	}
	vector<int> topKFrequent(vector<int>& nums, int k) {
		//注意小顶堆优先队列的自定义排序写法，第三个参数 decltype（&cmp） s（cmp）
		priority_queue<vector<int>, vector<vector<int>>, decltype(&cmp)> s(cmp);
		map<int, int>m;

		for (int i : nums) {
			m[i]++;
		}
		for (auto[key, val] : m) {
			if (s.size() == k) {
				if (s.top()[1] < val) {
					s.pop();
					s.push({ key,val });
				}
			}
			else s.push({ key,val });
		}
		vector<int> ans;
		while (!s.empty()) {
			ans.push_back(s.top()[0]);
			s.pop();
		}
		return ans;
	}
};


//求和
/*
在不考虑进位的情况下，其无进位加法结果为 a⊕b。

而所有需要进位的位为 a & b，进位后的进位结果为 (a & b) << 1。

*/
int getSum(int a, int b) {
	while (b != 0) {
		unsigned int carr = (unsigned int)(a&b) << 1;
		a = a ^ b;
		b = carr;
	}
	return a;
}

//矩阵中第k小的元素
class Solution378 {
public:
	bool check(vector<vector<int>>& matrix, int k, int mid) {
		//matrix中小于等于mid的数是否小于k
		int count = 0;
		int r = matrix.size(), c = matrix[0].size();
		for (int i = 0, j = c - 1; i < r&&j >= 0;) {
			if (matrix[i][j] > mid) {
				j--;
			}
			else {
				i++;
				count = count + (j + 1);
			}
		}
		return count < k;
	}
	int kthSmallest(vector<vector<int>>& matrix, int k) {
		int l = matrix[0][0];
		int r = matrix[matrix.size() - 1].back();
		while (l < r) {
			int mid = l + (r - l) / 2;
			//matrix中小于等于mid的数 如果小于k mid要舍弃掉也即是mid++
			if (check(matrix, k, mid))l = mid + 1;
			//否则 mid有可能是答案 不能舍弃 也就是 r=mid
			else r = mid;
		}
		return l;
	}
};

//打乱数组  洗牌算法
/*
设待原地乱序的数组 nums。
循环 n 次，在第 i 次循环中（0≤i<n）：
在 [i,n)中随机抽取一个下标 j；
将第 i 个元素与第 j 个元素交换。

*/
class Solution384 {
private:
	vector<int> nums;
	vector<int> original;
public:
	Solution384(vector<int>& nums) {
		this->nums = nums;
		this->original.resize(nums.size());
		//copy只负责复制，不负责申请空间，所以复制前必须有足够的空间
		copy(nums.begin(), nums.end(), original.begin());
	}

	vector<int> reset() {
		return original;
	}

	vector<int> shuffle() {
		for (int i = 0; i < nums.size(); ++i) {
			int j = i + rand() % (nums.size() - i);
			swap(nums[i], nums[j]);
		}
		return nums;
	}
};


//395 至少有k个重复字符的最长字串
/*
对于字符串 s，如果存在某个字符 ch，它的出现次数大于 0 且小于 k，则任何包含 ch 的子串都不可能满足要求。
也就是说，我们将字符串按照 ch 切分成若干段，则满足要求的最长子串一定出现在某个被切分的段内，而不能跨越一个或多个段。
*/
int longestSubstring(string s, int k) {
	int n = s.size();
	if (k < 2) return n;
	if (n < k) return 0;
	int m[26] = { 0 };
	for (auto c : s) ++m[c - 'a'];
	int i = 0;
	//找到第一个出现次数不到k的字符
	while (i < n && m[s[i] - 'a'] >= k) ++i;
	if (i == n) return n;
	//递归求左边的满足要求的长度
	int left = longestSubstring(s.substr(0, i), k);
	//出现次数小于k的肯定不在满足要求的子串中，跳过
	while (i < n && m[s[i] - 'a'] < k) ++i;
	//递归求右边的满足要求的长度
	int right = longestSubstring(s.substr(i), k);
	return max(left, right);
}

//4数之和 454  分组 + 哈希
int fourSumCount(vector<int>& nums1, vector<int>& nums2, vector<int>& nums3, vector<int>& nums4) {
	unordered_map<int, int>m;
	for (int i : nums1) {
		for (int j : nums2)m[i + j]++;
	}
	int ans = 0;
	for (int i : nums3) {
		for (int j : nums4) {
			if (m.count(-i - j))ans = ans + m[-i - j];
		}
	}
	return ans;
}
int main()
{
    std::cout << "Hello World!\n";

	string s = "catsandog";
	vector<string> wordDict = { "cats", "dog", "sand", "and", "cat" };
	vector<int> nums = { 20,100,10,12,5,13};
	increasingTriplet(nums);
	std::cout << "Hello World!\n";
	std::cout << "Hello World!\n";


}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
