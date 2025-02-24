#include "Top100Liked.h"

/*
����һ���������� nums ��һ������Ŀ��ֵ target�������ڸ��������ҳ� ��ΪĿ��ֵ target  ���� ���� ���������������ǵ������±ꡣ
����Լ���ÿ������ֻ���Ӧһ���𰸣������㲻��ʹ��������ͬ��Ԫ�ء�
����԰�����˳�򷵻ش𰸡�

ʾ�� 1��
���룺nums = [2,7,11,15], target = 9
�����[0,1]
���ͣ���Ϊ nums[0] + nums[1] == 9 ������ [0, 1] ��

ʾ�� 2��
���룺nums = [3,2,4], target = 6
�����[1,2]

ʾ�� 3��
���룺nums = [3,3], target = 6
�����[0,1]

��ʾ��
2 <= nums.length <= 104
-109 <= nums[i] <= 109
-109 <= target <= 109
ֻ�����һ����Ч��

���ף���������һ��ʱ�临�Ӷ�С�� O(n2) ���㷨��
*/
vector<int> Top100Liked::Solution::twoSum(vector<int>& nums, int target, bool isMyOwn)
{
	if (isMyOwn) {
		int i, j;
		for (i = 0; i < nums.size() - 1; i++)
		{
			for (j = i + 1; j < nums.size(); j++)
			{
				if (nums[i] + nums[j] == target)
				{
					return { i,j };
				}
			}
		}
		return { };
	}
	else {
		unordered_map<int, int> hashtable;
		for (int i = 0; i < nums.size(); ++i) {
			auto it = hashtable.find(target - nums[i]);
			if (it != hashtable.end()) {
				return { it->second, i };
			}
			hashtable[nums[i]] = i;
		}
		return {};
	}
}

/*
����һ���ַ������飬���㽫 ��ĸ��λ�� �����һ�𡣿��԰�����˳�򷵻ؽ���б�
��ĸ��λ�� ������������Դ���ʵ�������ĸ�õ���һ���µ��ʡ�

ʾ�� 1:
����: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
���: [["bat"],["nat","tan"],["ate","eat","tea"]]

ʾ�� 2:
����: strs = [""]
���: [[""]]

ʾ�� 3:
����: strs = ["a"]
���: [["a"]]

��ʾ��
1 <= strs.length <= 104
0 <= strs[i].length <= 100
strs[i] ������Сд��ĸ
*/
vector<vector<string>> Top100Liked::Solution::groupAnagrams(vector<string>& strs, bool isMyOwn)
{
	if (isMyOwn) {
		unordered_map<string, vector<string>> mp;
		vector<vector<string>> result;
		for (auto& i : strs)
		{
			string sKey = i;
			sort(sKey.begin(), sKey.end());
			mp[sKey].emplace_back(i);
		}
		for (auto i : mp)
		{
			result.emplace_back(i.second);
		}
		return result;
	}
	else {
		unordered_map<string, vector<string>> mp;
		for (string& str : strs) {
			string key = str;
			sort(key.begin(), key.end());
			mp[key].emplace_back(str);
		}
		vector<vector<string>> ans;
		for (auto it = mp.begin(); it != mp.end(); ++it) {
			ans.emplace_back(it->second);
		}
		return ans;
	}
}

/*
����һ��δ������������� nums ���ҳ���������������У���Ҫ������Ԫ����ԭ�������������ĳ��ȡ�
������Ʋ�ʵ��ʱ�临�Ӷ�Ϊ O(n) ���㷨��������⡣

ʾ�� 1��
���룺nums = [100,4,200,1,3,2]
�����4
���ͣ���������������� [1, 2, 3, 4]�����ĳ���Ϊ 4��

ʾ�� 2��
���룺nums = [0,3,7,2,5,8,4,6,0,1]
�����9

ʾ�� 3��
���룺nums = [1,0,1,2]
�����3

��ʾ��
0 <= nums.length <= 105
-109 <= nums[i] <= 109
*/
int Top100Liked::Solution::longestConsecutive(vector<int>& nums, bool isMyOwn)
{
	if (isMyOwn) {
		unordered_set<int> set;
		for (const auto& num : nums) {
			set.insert(num);
		}
		int ret = set.size() > 0 ? 1 : 0;
		for (auto num : set) {
			if (set.find(num - 1) != set.end())
			{
				continue;
			}
			int count = 1;
			while (set.find(num + count) != set.end())
			{
				count++;
			}
			if (count > ret)
			{
				ret = count;
			}
		}
		return ret;
	}
	else {
		unordered_set<int> num_set;
		for (const int& num : nums) {
			num_set.insert(num);
		}
		int longestStreak = 0;
		for (const int& num : num_set) {
			if (!num_set.count(num - 1)) {
				int currentNum = num;
				int currentStreak = 1;

				while (num_set.count(currentNum + 1)) {
					currentNum += 1;
					currentStreak += 1;
				}

				longestStreak = max(longestStreak, currentStreak);
			}
		}
		return longestStreak;
	}
}

/*
����һ������ nums����дһ������������ 0 �ƶ��������ĩβ��ͬʱ���ַ���Ԫ�ص����˳��
��ע�� �������ڲ���������������ԭ�ض�������в�����

ʾ�� 1:
����: nums = [0,1,0,3,12]
���: [1,3,12,0,0]

ʾ�� 2:
����: nums = [0]
���: [0]

��ʾ:
1 <= nums.length <= 104
-2��31�η� <= nums[i] <= 2��31�η� - 1

���ף����ܾ���������ɵĲ���������
*/
void Top100Liked::Solution::moveZeroes(vector<int>& nums, bool isMyOwn)
{
	if (isMyOwn) {
		int num;
		for (int i = 0, j = 0; i < nums.size(); i++)
		{
			if (nums[i] != 0)
			{
				num = nums[j];
				nums[j] = nums[i];
				nums[i] = num;
				j++;
			}
		}
	}
	else {
		int n = nums.size(), left = 0, right = 0;
		while (right < n) {
			if (nums[right]) {
				swap(nums[left], nums[right]);
				left++;
			}
			right++;
		}
	}
}

/*
����һ������Ϊ n ���������� height ���� n �����ߣ��� i ���ߵ������˵��� (i, 0) �� (i, height[i]) ��
�ҳ����е������ߣ�ʹ�������� x �Ṳͬ���ɵ�����������������ˮ��
�����������Դ�������ˮ����
˵�����㲻����б������

ʾ�� 1��
���룺[1,8,6,2,5,4,8,3,7]
�����49
���ͣ�ͼ�д�ֱ�ߴ����������� [1,8,6,2,5,4,8,3,7]���ڴ�����£������ܹ�����ˮ����ʾΪ��ɫ���֣������ֵΪ 49��

ʾ�� 2��
���룺height = [1,1]
�����1

��ʾ��
n == height.length
2 <= n <= 105
0 <= height[i] <= 104
*/
int Top100Liked::Solution::maxArea(vector<int>& height, bool isMyOwn)
{
	if (isMyOwn) {
		int ret = 0, i = 0, j = height.size() - 1;
		while (i < j)
		{
			bool direct = height[i] > height[j];
			int area = (j - i) * (direct ? height[j] : height[i]);
			ret = area > ret ? area : ret;
			if (direct)
			{
				j--;
			}
			else
			{
				i++;
			}
		}
		return ret;
	}
	else {
		int l = 0, r = height.size() - 1;
		int ans = 0;
		while (l < r) {
			int area = min(height[l], height[r]) * (r - l);
			ans = max(ans, area);
			if (height[l] <= height[r]) {
				++l;
			}
			else {
				--r;
			}
		}
		return ans;
	}
}

/*
����һ���������� nums ���ж��Ƿ������Ԫ�� [nums[i], nums[j], nums[k]] ���� i != j��i != k �� j != k ��ͬʱ������ nums[i] + nums[j] + nums[k] == 0 �����㷵�����к�Ϊ 0 �Ҳ��ظ�����Ԫ�顣
ע�⣺���в����԰����ظ�����Ԫ�顣

ʾ�� 1��
���룺nums = [-1,0,1,2,-1,-4]
�����[[-1,-1,2],[-1,0,1]]
���ͣ�
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0 ��
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0 ��
nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0 ��
��ͬ����Ԫ���� [-1,0,1] �� [-1,-1,2] ��
ע�⣬�����˳�����Ԫ���˳�򲢲���Ҫ��

ʾ�� 2��
���룺nums = [0,1,1]
�����[]
���ͣ�Ψһ���ܵ���Ԫ��Ͳ�Ϊ 0 ��

ʾ�� 3��
���룺nums = [0,0,0]
�����[[0,0,0]]
���ͣ�Ψһ���ܵ���Ԫ���Ϊ 0 ��

��ʾ��
3 <= nums.length <= 3000
-105 <= nums[i] <= 105
*/
vector<vector<int>> Top100Liked::Solution::threeSum(vector<int>& nums, bool isMyOwn)
{
	isMyOwn = false;
	if (isMyOwn) {
		int n = nums.size();
		sort(nums.begin(), nums.end());
		vector<vector<int>> ans;

		//ѭ������ߵ�һ����
		for (int first = 0; first < n; ++first)
		{
			// ʹ����һ�ε�������һ�β�ͬ
			if (first > 0 && nums[first] == nums[first - 1])
			{
				continue;
			}
			// �����������������ұ�
			int third = n - 1;
			// a+b+c=0 ת���� b+c=-a;
			int target = -nums[first];
			// ѭ����ߵڶ��������ڵ�һ��������
			for (int second = first + 1; second < n; ++second)
			{
				// ʹ����һ�ε�������һ�β�ͬ
				if (second > first + 1 && nums[second] == nums[second - 1])
				{
					continue;
				}
				// �õ�������һֱ�ڵڶ������ұ�
				while (second < third && nums[second] + nums[third] > target)
				{
					--third;
				}
				//���غϺ�ͱ�������b��c �˳���һ��ѭ��
				if (second == third)
				{
					break;
				}
				if (nums[second] + nums[third] == target)
				{
					ans.push_back({ nums[first], nums[second], nums[third] });
				}
			}
		}
		return ans;
	}
	else {
		int n = nums.size();
		sort(nums.begin(), nums.end());
		vector<vector<int>> ans;
		// ö�� a
		for (int first = 0; first < n; ++first) {
			// ��Ҫ����һ��ö�ٵ�������ͬ
			if (first > 0 && nums[first] == nums[first - 1]) {
				continue;
			}
			// c ��Ӧ��ָ���ʼָ����������Ҷ�
			int third = n - 1;
			int target = -nums[first];
			// ö�� b
			for (int second = first + 1; second < n; ++second) {
				// ��Ҫ����һ��ö�ٵ�������ͬ
				if (second > first + 1 && nums[second] == nums[second - 1]) {
					continue;
				}
				// ��Ҫ��֤ b ��ָ���� c ��ָ������
				while (second < third && nums[second] + nums[third] > target) {
					--third;
				}
				// ���ָ���غϣ����� b ����������
				// �Ͳ��������� a+b+c=0 ���� b<c �� c �ˣ������˳�ѭ��
				if (second == third) {
					break;
				}
				if (nums[second] + nums[third] == target) {
					ans.push_back({ nums[first], nums[second], nums[third] });
				}
			}
		}
		return ans;
	}
}

/*
���� n ���Ǹ�������ʾÿ�����Ϊ 1 �����ӵĸ߶�ͼ�����㰴�����е����ӣ�����֮���ܽӶ�����ˮ��

ʾ�� 1��
���룺height = [0,1,0,2,1,0,1,3,2,1,2,1]
�����6
���ͣ������������� [0,1,0,2,1,0,1,3,2,1,2,1] ��ʾ�ĸ߶�ͼ������������£����Խ� 6 ����λ����ˮ����ɫ���ֱ�ʾ��ˮ����

ʾ�� 2��
���룺height = [4,2,0,3,2,5]
�����9

��ʾ��
n == height.length
1 <= n <= 2 * 104
0 <= height[i] <= 105
*/
int Top100Liked::Solution::trap(vector<int>& height, bool isMyOwn)
{
	if (isMyOwn) {
		int maxContain = 0;

		int leftH = 0, rightH = 0;
		int left = 0, right = height.size() - 1;
		while (left < right)
		{
			leftH = max(leftH, height[left]);
			rightH = max(rightH, height[right]);
			if (leftH < rightH)
			{
				maxContain += leftH - height[left];
				left++;
			}
			else
			{
				maxContain += rightH - height[right];
				right--;
			}
		}

		return maxContain;
	}
	else {
		int ans = 0;
		int left = 0, right = height.size() - 1;
		int leftMax = 0, rightMax = 0;
		while (left < right) {
			leftMax = max(leftMax, height[left]);
			rightMax = max(rightMax, height[right]);
			if (height[left] < height[right]) {
				ans += leftMax - height[left];
				++left;
			}
			else {
				ans += rightMax - height[right];
				--right;
			}
		}
		return ans;
	}
}

/*
����һ���ַ��� s �������ҳ����в������ظ��ַ��� ��Ӵ��ĳ��ȡ�

ʾ�� 1:
����: s = "abcabcbb"
���: 3
����: ��Ϊ���ظ��ַ�����Ӵ��� "abc"�������䳤��Ϊ 3��

ʾ�� 2:
����: s = "bbbbb"
���: 1
����: ��Ϊ���ظ��ַ�����Ӵ��� "b"�������䳤��Ϊ 1��

ʾ�� 3:
����: s = "pwwkew"
���: 3
����: ��Ϊ���ظ��ַ�����Ӵ��� "wke"�������䳤��Ϊ 3��
	 ��ע�⣬��Ĵ𰸱����� �Ӵ� �ĳ��ȣ�"pwke" ��һ�������У������Ӵ���

��ʾ��
0 <= s.length <= 5 * 104
s ��Ӣ����ĸ�����֡����źͿո����
*/
int Top100Liked::Solution::lengthOfLongestSubstring(string s, bool isMyOwn)
{
	if (isMyOwn) {
		if (s.length() <= 1) { return s.length(); }
		std::string maxStr = "";
		size_t current = 1;
		size_t maxLength = 1;
		maxStr.push_back(s[0]);
		for (size_t i = 0; i < s.length() - 1; i++)
		{
			for (current; current < s.length(); current++)
			{
				if (maxStr.find(s[current]) != -1)
				{
					break;
				}
				else {
					maxStr.push_back(s[current]);
				}
			}
			maxLength = maxStr.length() > maxLength ? maxStr.length() : maxLength;
			maxStr = maxStr.substr(1);
		}
		return maxLength;
	}
	else {
		// ��ϣ���ϣ���¼ÿ���ַ��Ƿ���ֹ�
		unordered_set<char> occ;
		int n = s.size();
		// ��ָ�룬��ʼֵΪ -1���൱���������ַ�������߽����࣬��û�п�ʼ�ƶ�
		int rk = -1, ans = 0;
		// ö����ָ���λ�ã���ʼֵ���Եر�ʾΪ -1
		for (int i = 0; i < n; ++i) {
			if (i != 0) {
				// ��ָ�������ƶ�һ���Ƴ�һ���ַ�
				occ.erase(s[i - 1]);
			}
			while (rk + 1 < n && !occ.count(s[rk + 1])) {
				// ���ϵ��ƶ���ָ��
				occ.insert(s[rk + 1]);
				++rk;
			}
			// �� i �� rk ���ַ���һ�����������ظ��ַ��Ӵ�
			ans = max(ans, rk - i + 1);
		}
		return ans;
	}
}

/*
���������ַ��� s �� p���ҵ� s ������ p ����λ�ʵ��Ӵ���������Щ�Ӵ�����ʼ�����������Ǵ������˳��

ʾ�� 1:
����: s = "cbaebabacd", p = "abc"
���: [0,6]
����:
��ʼ�������� 0 ���Ӵ��� "cba", ���� "abc" ����λ�ʡ�
��ʼ�������� 6 ���Ӵ��� "bac", ���� "abc" ����λ�ʡ�

ʾ�� 2:
����: s = "abab", p = "ab"
���: [0,1,2]
����:
��ʼ�������� 0 ���Ӵ��� "ab", ���� "ab" ����λ�ʡ�
��ʼ�������� 1 ���Ӵ��� "ba", ���� "ab" ����λ�ʡ�
��ʼ�������� 2 ���Ӵ��� "ab", ���� "ab" ����λ�ʡ�

��ʾ:
1 <= s.length, p.length <= 3 * 104
s �� p ������Сд��ĸ

��ע:
differ��ʾ�ж��ٸ���ĸ��ͬ��ֻҪcount�д���һ����Ϊ0����ĸ���������һ����ͬ�����ڳ���һֱΪsLen-pLen����������Ƴ���ĸ��ԭ����Ϊ1���������1��Ϊ0������ͬ��1��ͬ���Ҳ��������ĸΪ-1��������Ϊ0����ͬ��1��ͬ��differ��һҲ������ԭ��
*/
vector<int> Top100Liked::Solution::findAnagrams(string s, string p, bool isMyOwn)
{
	if (isMyOwn) {
		vector<int> result;
		int left = 0, right = 0;
		string temp;
		bool isFind;
		for (left; left < s.length(); left++)
		{
			isFind = false;
			temp = p;
			if (s.substr(left, p.length()).find(p) != -1)
			{
				isFind = true;
			}
			else {
				for (right = left; right < s.length(); right++)
				{
					int findIndex = temp.find(s[right]);
					if (findIndex != -1)
					{
						temp = temp.substr(0, findIndex) + temp.substr(findIndex + 1);
						if (temp.length() == 0)
						{
							isFind = true;
						}
					}
					else
					{
						break;
					}
				}
			}
			if (isFind)
			{
				result.push_back(left);
			}
		}
		return result;
	}
	else {
		int sLen = s.size(), pLen = p.size();
		if (sLen < pLen) {
			return vector<int>();
		}
		vector<int> ans;
		vector<int> count(26);
		for (int i = 0; i < pLen; ++i) {
			++count[s[i] - 'a'];
			--count[p[i] - 'a'];
		}
		int differ = 0;
		for (int j = 0; j < 26; ++j) {
			if (count[j] != 0) {
				++differ;
			}
		}
		if (differ == 0) {
			ans.emplace_back(0);
		}
		for (int i = 0; i < sLen - pLen; ++i) {
			if (count[s[i] - 'a'] == 1) {  // ��������ĸ s[i] ���������ַ��� p �е������Ӳ�ͬ�����ͬ
				--differ;
			}
			else if (count[s[i] - 'a'] == 0) {  // ��������ĸ s[i] ���������ַ��� p �е���������ͬ��ò�ͬ
				++differ;
			}
			--count[s[i] - 'a'];

			if (count[s[i + pLen] - 'a'] == -1) {  // ��������ĸ s[i+pLen] ���������ַ��� p �е������Ӳ�ͬ�����ͬ
				--differ;
			}
			else if (count[s[i + pLen] - 'a'] == 0) {  // ��������ĸ s[i+pLen] ���������ַ��� p �е���������ͬ��ò�ͬ
				++differ;
			}
			++count[s[i + pLen] - 'a'];

			if (differ == 0) {
				ans.emplace_back(i + 1);
			}
		}
		return ans;
	}
}

/*
����һ���������� nums ��һ������ k ������ͳ�Ʋ����� �������к�Ϊ k ��������ĸ��� ��
��������������Ԫ�ص������ǿ����С�

ʾ�� 1��
���룺nums = [1,1,1], k = 2
�����2

ʾ�� 2��
���룺nums = [1,2,3], k = 3
�����2

��ʾ��
1 <= nums.length <= 2 * 104
-1000 <= nums[i] <= 1000
-107 <= k <= 107

��ע:
����preΪǰ���ۼ��ܺͣ�ͨ��sum[i-j]��λ���м��Ϊk�Ļ�����ôsum[0-j]-sum[0,i]����sum[i-j]��Ҳ����kֵ��Ȼ���0-n�������ۼ�ֵ�����ֹ�������ֻҪ��ֵΪk����ǰ���sum[0-i]���ֹ����ɡ�
*/
int Top100Liked::Solution::subarraySum(vector<int>& nums, int k, bool isMyOwn)
{
	isMyOwn = false;
	if (isMyOwn) {
		unordered_map<int, int> map;
		if (nums.size() != 0) {
			for (int i = 0; i < nums.size(); i++)
			{
				int sum = 0;
				int left = i;
				for (int right = left; right < nums.size(); right++)
				{
					if (sum < k) {
						sum += nums[right];
					}
					else if (sum > k) {
						sum -= nums[left];
						left++;
						right--;
					}
					else {
						sum += nums[right];
						sum -= nums[left];
						left++;
					}
					if (sum == k) {
						map[left + right]++;
					}
				}
			}
		}
		return map.size();
	}
	else {
		unordered_map<int, int> mp;
		mp[0] = 1;
		int count = 0, pre = 0;
		for (auto& x : nums) {
			pre += x;
			if (mp.find(pre - k) != mp.end()) {
				count += mp[pre - k];
			}
			mp[pre]++;
		}
		return count;
	}
}

/*
����һ���������� nums����һ����СΪ k �Ļ������ڴ������������ƶ�����������Ҳࡣ��ֻ���Կ����ڻ��������ڵ� k �����֡���������ÿ��ֻ�����ƶ�һλ��
���� ���������е����ֵ ��

ʾ�� 1��
���룺nums = [1,3,-1,-3,5,3,6,7], k = 3
�����[3,3,5,5,6,7]
���ͣ�
�������ڵ�λ��                ���ֵ
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7

ʾ�� 2��
���룺nums = [1], k = 1
�����[1]

ʾ�� 3��
���룺nums = [1,3,1,2,0,5], k = 3
�����[3,3,2,5]

ʾ�� 4��
���룺nums = [-7,-8,7,5,7,1,6,0], k = 4
�����[7,7,7,7,7]

��ʾ��
1 <= nums.length <= 105
-104 <= nums[i] <= 104
1 <= k <= nums.length

��ע��
��ԭ���Ǹ���k���ֶΣ��ȴ������ң�ÿ���ڳ����Ƚϣ��������ֵ������С����Ϊp��ͬ��������󣬳����Ƚ��������ֵ��Ҳ�Ǵ�С����Ϊs��
���պ���k��С��ʱ����p���Ҳ��s����඼�����ֵ������Ϊk�ı���ʱ����������k���ȷ��飬��ô����Ҫ�Ƚ�p��Χ�����Ҳ�����ֵ��s��Χ�����������ֵ���õ����ֵ��
ͬʱ��Ҫ���ǳ��Ȳ��Ǹպ�Ϊk�ı��������������ʼֵ[0]��[n-1]��ֵ����ʾ�������������˸�����Ŀ��ǡ�
*/
vector<int> Top100Liked::Solution::maxSlidingWindow(vector<int>& nums, int k, bool isMyOwn)
{
	if (isMyOwn) {
		if (nums.size() == 0) { return vector<int>(); }
		vector<int> result;
		int i = 0;
		multiset<int> set;
		for (i; i < k; i++) {
			set.insert(nums.at(i));
		}
		if (set.size() > 0) {
			result.push_back(*set.rbegin());
		}
		int left = 0;
		for (i; i < nums.size(); i++, left++) {
			set.erase(set.find(nums.at(left)));
			set.insert(nums.at(i));
			result.push_back(*set.rbegin());
		}
		return result;
	}
	else {
		int n = nums.size();
		vector<int> prefixMax(n), suffixMax(n);
		for (int i = 0; i < n; ++i) {
			if (i % k == 0) {
				prefixMax[i] = nums[i];
			}
			else {
				prefixMax[i] = max(prefixMax[i - 1], nums[i]);
			}
		}
		for (int i = n - 1; i >= 0; --i) {
			if (i == n - 1 || (i + 1) % k == 0) {
				suffixMax[i] = nums[i];
			}
			else {
				suffixMax[i] = max(suffixMax[i + 1], nums[i]);
			}
		}
		vector<int> ans;
		for (int i = 0; i <= n - k; ++i) {
			ans.push_back(max(suffixMax[i], prefixMax[i + k - 1]));
		}
		return ans;
	}
}

bool check(unordered_map <char, int> ori, unordered_map <char, int> cnt) {
	for (const auto p : ori) {
		if (cnt[p.first] < p.second) {
			return false;
		}
	}
	return true;
}
/*
����һ���ַ��� s ��һ���ַ��� t ������ s �к��� t �����ַ�����С�Ӵ������ s �в����ں��� t �����ַ����Ӵ����򷵻ؿ��ַ��� "" ��

ע�⣺
���� t ���ظ��ַ�������Ѱ�ҵ����ַ����и��ַ��������벻���� t �и��ַ�������
��� s �д����������Ӵ������Ǳ�֤����Ψһ�Ĵ𰸡�

ʾ�� 1��
���룺s = "ADOBECODEBANC", t = "ABC"
�����"BANC"
���ͣ���С�����Ӵ� "BANC" ���������ַ��� t �� 'A'��'B' �� 'C'��

ʾ�� 2��
���룺s = "a", t = "a"
�����"a"
���ͣ������ַ��� s ����С�����Ӵ���

ʾ�� 3:
����: s = "a", t = "aa"
���: ""
����: t �������ַ� 'a' ��Ӧ������ s ���Ӵ��У�
���û�з������������ַ��������ؿ��ַ�����

��ʾ��
m == s.length
n == t.length
1 <= m, n <= 105
s �� t ��Ӣ����ĸ���

���ף��������һ���� o(m+n) ʱ���ڽ����������㷨��
*/
string Top100Liked::Solution::minWindow(string s, string t, bool isMyOwn)
{
	if (isMyOwn) {
		if (t.length() > s.length() || t.length() == 0) { return string(); }
		unordered_map<char, int> map;
		for (auto c : t) {
			map[c]++;
		}
		int diff = map.size();
		int left = 0, right = 0, start = -1, length = s.length() + 1;
		for (right; right < s.length(); right++) {
			if (--map[s[right]] == 0) {
				diff--;
			}
			while (diff == 0 && left <= right) {
				if ((right - left + 1) < length) {
					length = right - left + 1;
					start = left;
				}
				if (++map[s[left]] > 0) {
					diff++;
				}
				left++;
			}
		}
		if (start == -1 || length == s.length() + 1) {
			return string();
		}
		else {
			return s.substr(start, length);
		}
	}
	else {
		unordered_map <char, int> ori, cnt;
		for (const auto& c : t) {
			++ori[c];
		}
		int l = 0, r = -1;
		int len = INT_MAX, ansL = -1, ansR = -1;
		while (r < int(s.size())) {
			if (ori.find(s[++r]) != ori.end()) {
				++cnt[s[r]];
			}
			while (check(ori, cnt) && l <= r) {
				if (r - l + 1 < len) {
					len = r - l + 1;
					ansL = l;
				}
				if (ori.find(s[l]) != ori.end()) {
					--cnt[s[l]];
				}
				++l;
			}
		}
		return ansL == -1 ? string() : s.substr(ansL, len);
	}
}

struct Status {
	int lSum, rSum, mSum, iSum;
	Status(int ls, int rs, int ms, int is)
		: lSum(ls), rSum(rs), mSum(ms), iSum(is) {}
};
Status pushUp(Status l, Status r) {
	int iSum = l.iSum + r.iSum;
	int lSum = max(l.lSum, l.iSum + r.lSum);
	int rSum = max(r.rSum, r.iSum + l.rSum);
	int mSum = max(max(l.mSum, r.mSum), l.rSum + r.lSum);
	return Status(lSum, rSum, mSum, iSum);
};
Status get(vector<int>& a, int l, int r) {
	if (l == r) {
		return Status(a[l], a[l], a[l], a[l]);
	}
	int m = (l + r) >> 1;
	Status lSub = get(a, l, m);
	Status rSub = get(a, m + 1, r);
	return pushUp(lSub, rSub);
}
/*
����һ���������� nums �������ҳ�һ���������͵����������飨���������ٰ���һ��Ԫ�أ������������͡�
�������������е�һ���������֡�

ʾ�� 1��
���룺nums = [-2,1,-3,4,-1,2,1,-5,4]
�����6
���ͣ����������� [4,-1,2,1] �ĺ����Ϊ 6 ��

ʾ�� 2��
���룺nums = [1]
�����1

ʾ�� 3��
���룺nums = [5,4,-1,7,8]
�����23

��ʾ��
1 <= nums.length <= 105
-104 <= nums[i] <= 104

���ף�������Ѿ�ʵ�ָ��Ӷ�Ϊ O(n) �Ľⷨ������ʹ�ø�Ϊ����� ���η� ��⡣
*/
int Top100Liked::Solution::maxSubArray(vector<int>& nums, bool isMyOwn)
{
	if (isMyOwn) {
		int numsSize = nums.size();
		int result = nums[0];
		int pTotal = nums[0];
		for (int i = 1; i < numsSize; i++)
		{
			pTotal = max(pTotal + nums[i], nums[i]);
			result = max(result, pTotal);
		}
		return result;
	}
	else {
		return get(nums, 0, nums.size() - 1).mSum;
	}
}

/*
������ intervals ��ʾ���ɸ�����ļ��ϣ����е�������Ϊ intervals[i] = [starti, endi] ������ϲ������ص������䣬������ һ�����ص����������飬��������ǡ�ø��������е��������� ��

ʾ�� 1��
���룺intervals = [[1,3],[2,6],[8,10],[15,18]]
�����[[1,6],[8,10],[15,18]]
���ͣ����� [1,3] �� [2,6] �ص�, �����Ǻϲ�Ϊ [1,6].

ʾ�� 2��
���룺intervals = [[1,4],[4,5]]
�����[[1,5]]
���ͣ����� [1,4] �� [4,5] �ɱ���Ϊ�ص����䡣

ʾ�� 3��
���룺intervals = [[2,5],[1,5]]
�����[[1,5]]

ʾ�� 4��
���룺intervals = [[1,5],[2,3]]
�����[[1,5]]

��ʾ��
1 <= intervals.length <= 104
intervals[i].length == 2
0 <= starti <= endi <= 104

��ע��
�����Ѿ��������򣬴�С����ıȽ���Լ򵥴���������Ҫ��������������Ƿ����β�������Ҳ༴�ɡ�
*/
vector<vector<int>> Top100Liked::Solution::merge(vector<vector<int>>& intervals, bool isMyOwn)
{
	if (isMyOwn) {
		sort(intervals.begin(), intervals.end());
		vector<vector<int>> result;
		for (auto vec : intervals) {
			if (!result.empty() && (result.back()[1] >= vec[0] || result.back()[1] >= vec[1])) {
				int left = result.back()[0];
				int right = result.back()[1] > vec[1] ? result.back()[1] : vec[1];
				result.pop_back();
				result.push_back({ left, right });
			}
			else {
				result.push_back(vec);
			}
		}
		return result;
	}
	else {
		if (intervals.size() == 0) {
			return {};
		}
		sort(intervals.begin(), intervals.end());
		vector<vector<int>> merged;
		for (int i = 0; i < intervals.size(); ++i) {
			int L = intervals[i][0], R = intervals[i][1];
			if (!merged.size() || merged.back()[1] < L) {
				merged.push_back({ L, R });
			}
			else {
				merged.back()[1] = max(merged.back()[1], R);
			}
		}
		return merged;
	}
}

/*
����һ���������� nums���������е�Ԫ��������ת k ��λ�ã����� k �ǷǸ�����

ʾ�� 1:
����: nums = [1,2,3,4,5,6,7], k = 3
���: [5,6,7,1,2,3,4]
����:
������ת 1 ��: [7,1,2,3,4,5,6]
������ת 2 ��: [6,7,1,2,3,4,5]
������ת 3 ��: [5,6,7,1,2,3,4]

ʾ�� 2:
���룺nums = [-1,-100,3,99], k = 2
�����[3,99,-1,-100]
����:
������ת 1 ��: [99,-1,-100,3]
������ת 2 ��: [3,99,-1,-100]

��ʾ��
1 <= nums.length <= 105
-2��31�η� <= nums[i] <= 2��31�η� - 1
0 <= k <= 105

���ף�
�������������Ľ�������������� ���� ��ͬ�ķ������Խ��������⡣
�����ʹ�ÿռ临�Ӷ�Ϊ O(1) �� ԭ�� �㷨������������

��ע��
��Ϊ�ǰ���˳����Ҳ��Ƴ�������ӵ���࣬�ȳ��Ƚ����൱�ڷ�ת��һ�Ρ�
���Կ����Ƚ��з�ת���������ȳ��Ƚ�������k������˳������Ƴ��������ݵ�˳���λ�ã��ٽ�ʣ�����ݷ�ת��ԭ˳���λ�ü��ɡ�
*/
void Top100Liked::Solution::rotate(vector<int>& nums, int k, bool isMyOwn)
{
	if (isMyOwn) {
		if (nums.size() <= 1) { return; }
		int sweap = 0;
		// 1,2,3,4,5,6,7
		int num = nums.size();
		// ��ȥ�����ѭ���ƶ�����
		k = k % num;
		// 7 6 5 4 3 2 1
		for (int i = 0; i < (num / 2); i++)
		{
			sweap = nums[num - 1 - i];
			nums[num - 1 - i] = nums[i];
			nums[i] = sweap;
		}
		// 5 6 7 4 3 2 1
		for (int i = 0; i < (k / 2); i++)
		{
			sweap = nums[k - 1 - i];
			nums[k - 1 - i] = nums[i];
			nums[i] = sweap;
		}
		// 5 6 7 1 2 3 4
		for (int i = 0; i < ((num - k) / 2); i++)
		{
			sweap = nums[num - 1 - i];
			nums[num - 1 - i] = nums[k + i];
			nums[k + i] = sweap;
		}
		// 5 6 7 1 2 3 4
	}
	else {
		k %= nums.size();
		reverse(nums.begin(), nums.end());
		reverse(nums.begin(), nums.begin() + k);
		reverse(nums.begin() + k, nums.end());

		// ��״�滻���ݲ��Ƽ�
		//int n = nums.size();
		//k = k % n;
		//// �������Լ��, (48, 18) => 6, (-24, 0) => 24
		//int count = gcd(k, n);
		//for (int start = 0; start < count; ++start) {
		//	int current = start;
		//	int prev = nums[start];
		//	do {
		//		int next = (current + k) % n;
		//		swap(nums[next], prev);
		//		current = next;
		//	} while (start != current);
		//}
	}
}

/*
����һ���������� nums������ ���� answer ������ answer[i] ���� nums �г� nums[i] ֮�������Ԫ�صĳ˻� ��
��Ŀ���� ��֤ ���� nums֮������Ԫ�ص�ȫ��ǰ׺Ԫ�غͺ�׺�ĳ˻�����  32 λ ������Χ�ڡ�
�� ��Ҫʹ�ó��������� O(n) ʱ�临�Ӷ�����ɴ��⡣

ʾ�� 1:
����: nums = [1,2,3,4]
���: [24,12,8,6]

ʾ�� 2:
����: nums = [-1,1,0,-3,3]
���: [0,0,9,0,0]

��ʾ��
2 <= nums.length <= 105
-30 <= nums[i] <= 30
���� ��֤ ���� answer[i] ��  32 λ ������Χ��

���ף�������� O(1) �Ķ���ռ临�Ӷ�����������Ŀ�𣿣� ���ڶԿռ临�Ӷȷ�����Ŀ�ģ�������� ������Ϊ ����ռ䡣��

��ע��
ʱ�临�Ӷ� O(2n) ���Խ��ƿ��� O(n)������ѭ����������Ҳ���Է�������
*/
vector<int> Top100Liked::Solution::productExceptSelf(vector<int>& nums, bool isMyOwn)
{
	if (isMyOwn) {
		vector<int> result(nums.size());
		int total = 1;
		for (int i = 0; i < nums.size(); ++i) {
			total *= nums[i];
			result[i] = total;
		}
		total = 1;
		for (int i = nums.size() - 1; i >= 0; i--) {
			if (i == 0) {
				result[i] = total;
			}
			else {
				result[i] = result[i - 1] * total;
			}
			total *= nums[i];
		}
		return result;
	}
	else {
		int length = nums.size();
		vector<int> answer(length);

		// answer[i] ��ʾ���� i �������Ԫ�صĳ˻�
		// ��Ϊ����Ϊ '0' ��Ԫ�����û��Ԫ�أ� ���� answer[0] = 1
		answer[0] = 1;
		for (int i = 1; i < length; i++) {
			answer[i] = nums[i - 1] * answer[i - 1];
		}

		// R Ϊ�Ҳ�����Ԫ�صĳ˻�
		// �տ�ʼ�ұ�û��Ԫ�أ����� R = 1
		int R = 1;
		for (int i = length - 1; i >= 0; i--) {
			// �������� i����ߵĳ˻�Ϊ answer[i]���ұߵĳ˻�Ϊ R
			answer[i] = answer[i] * R;
			// R ��Ҫ�����ұ����еĳ˻������Լ�����һ�����ʱ��Ҫ����ǰֵ�˵� R ��
			R *= nums[i];
		}
		return answer;
	}
}

/*
����һ��δ������������� nums �������ҳ�����û�г��ֵ���С����������
����ʵ��ʱ�临�Ӷ�Ϊ O(n) ����ֻʹ�ó����������ռ�Ľ��������

ʾ�� 1��
���룺nums = [1,2,0]
�����3
���ͣ���Χ [1,2] �е����ֶ��������С�

ʾ�� 2��
���룺nums = [3,4,-1,1]
�����2
���ͣ�1 �������У��� 2 û�С�

ʾ�� 3��
���룺nums = [7,8,9,11,12]
�����1
���ͣ���С������ 1 û�г��֡�

��ʾ��
1 <= nums.length <= 105
-2��31�η� <= nums[i] <= 2��31�η� - 1

��ע��
�ٷ�ʾ����ʽ�ǲ��ù�ϣ����߼�ʵ�֣���ϣ����Ҳ��һ�����飬��i���ڽǱ�i-1��λ�á�
���� 3 4 -1 1
��3���ڽǱ�2�Ϻ�-1 4 3 1 ������Ҫ�ٴμ��-1�������������������ٵ���
��4���ڽǱ�3�Ϻ�-1 1 3 4 �������ٴμ����1�������������Ҳ�������λ�ã��ٴε���
��1���ڽǱ�0�Ϻ�1 -1 3 4 �������ٴμ����-1�������������������ٵ���
����3 4 ��������λ�ã������ٵ���
*/
int Top100Liked::Solution::firstMissingPositive(vector<int>& nums, bool isMyOwn)
{
	if (isMyOwn) {
		sort(nums.begin(), nums.end());
		int minNum = 0;
		for (int i = 0; i < nums.size(); ++i) {
			if (nums[i] > 0 && nums[i] == minNum + 1) {
				minNum = nums[i];
			}
		}
		if (minNum == 0) {
			minNum = 1;
		}
		else {
			minNum++;
		}
		return minNum;
	}
	else {
		for (int i = 0; i < nums.size(); i++) {
			while (nums[i] > 0 && nums[i] <= nums.size() && nums[i] != nums[nums[i] - 1]) {
				swap(nums[i], nums[nums[i] - 1]);
			}
		}
		for (int i = 0; i < nums.size(); i++) {
			if (nums[i] != i + 1)
				return i + 1;
		}
		return nums.size() + 1;
	}
}

/*
����һ�� m x n �ľ������һ��Ԫ��Ϊ 0 �����������к��е�����Ԫ�ض���Ϊ 0 ����ʹ�� ԭ�� �㷨��

ʾ�� 1��
���룺matrix = [[1,1,1],[1,0,1],[1,1,1]]
�����[[1,0,1],[0,0,0],[1,0,1]]

ʾ�� 2��
���룺matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
�����[[0,0,0,0],[0,4,5,0],[0,3,1,0]]

��ʾ��
m == matrix.length
n == matrix[0].length
1 <= m, n <= 200
-2��31�η� <= matrix[i][j] <= 2��31�η� - 1

���ף�
һ��ֱ�۵Ľ��������ʹ��  O(mn) �Ķ���ռ䣬���Ⲣ����һ���õĽ��������
һ���򵥵ĸĽ�������ʹ�� O(m + n) �Ķ���ռ䣬������Ȼ������õĽ��������
�������һ����ʹ�ó����ռ�Ľ��������
*/
void Top100Liked::Solution::setZeroes(vector<vector<int>>& matrix, bool isMyOwn)
{
	if (isMyOwn) {
		int m = matrix.size();
		int n = matrix[0].size();
		vector<int> top(200, -1);
		vector<int> left(200, -1);
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				if (matrix[i][j] == 0) {
					top[j] = 0;
					left[i] = 0;
				}
			}
		}
		for (int i = 0; i < n; ++i) {
			if (top[i] == 0) {
				for (int j = 0; j < m; j++) {
					matrix[j][i] = 0;
				}
			}
		}
		for (int i = 0; i < m; ++i) {
			if (left[i] == 0) {
				for (int j = 0; j < n; j++) {
					matrix[i][j] = 0;
				}
			}
		}
	}
	else {
		int m = matrix.size();
		int n = matrix[0].size();
		int flag_col0 = false;
		for (int i = 0; i < m; i++) {
			if (!matrix[i][0]) {
				flag_col0 = true;
			}
			for (int j = 1; j < n; j++) {
				if (!matrix[i][j]) {
					matrix[i][0] = matrix[0][j] = 0;
				}
			}
		}
		for (int i = m - 1; i >= 0; i--) {
			for (int j = 1; j < n; j++) {
				if (!matrix[i][0] || !matrix[0][j]) {
					matrix[i][j] = 0;
				}
			}
			if (flag_col0) {
				matrix[i][0] = 0;
			}
		}
	}
}

/*
����һ�� m �� n �еľ��� matrix ���밴�� ˳ʱ������˳�� �����ؾ����е�����Ԫ�ء�

ʾ�� 1��
���룺matrix = [[1,2,3],[4,5,6],[7,8,9]]
�����[1,2,3,6,9,8,7,4,5]

ʾ�� 2��
���룺matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
�����[1,2,3,4,8,12,11,10,9,5,6,7]

��ʾ��
m == matrix.length
n == matrix[i].length
1 <= m, n <= 10
-100 <= matrix[i][j] <= 100

��ע��
����������������ǽ��ÿ���ƶ���ǽ�ڶ�������С��
*/
vector<int> Top100Liked::Solution::spiralOrder(vector<vector<int>>& matrix, bool isMyOwn)
{
	if (isMyOwn) {
		int m = matrix.size();
		int n = matrix[0].size();
		vector<int> result;
		int start = 0, top = 0, left = 0, right = n - 1, bottom = m - 1;
		while (true) {
			// �㼶����
			for (int i = left; i <= right; i++) {
				result.push_back(matrix[top][i]);
				start++;
			}
			top++;
			if (start >= m * n) { break; }

			// �㼶����
			for (int i = top; i <= bottom; i++) {
				result.push_back(matrix[i][right]);
				start++;
			}
			right--;
			if (start >= m * n) { break; }

			// �㼶����
			for (int i = right; i >= left; i--) {
				result.push_back(matrix[bottom][i]);
				start++;
			}
			bottom--;
			if (start >= m * n) { break; }

			// �㼶����
			for (int i = bottom; i >= top; i--) {
				result.push_back(matrix[i][left]);
				start++;
			}
			left++;
			if (start >= m * n) { break; }
		}
		return result;
	}
	else {
		if (matrix.size() == 0 || matrix[0].size() == 0) {
			return {};
		}

		int rows = matrix.size(), columns = matrix[0].size();
		vector<int> order;
		int left = 0, right = columns - 1, top = 0, bottom = rows - 1;
		while (left <= right && top <= bottom) {
			for (int column = left; column <= right; column++) {
				order.push_back(matrix[top][column]);
			}
			for (int row = top + 1; row <= bottom; row++) {
				order.push_back(matrix[row][right]);
			}
			if (left < right&& top < bottom) {
				for (int column = right - 1; column > left; column--) {
					order.push_back(matrix[bottom][column]);
				}
				for (int row = bottom; row > top; row--) {
					order.push_back(matrix[row][left]);
				}
			}
			left++;
			right--;
			top++;
			bottom--;
		}
		return order;
	}
}

/*
����һ�� n �� n �Ķ�ά���� matrix ��ʾһ��ͼ�����㽫ͼ��˳ʱ����ת 90 �ȡ�
������� ԭ�� ��תͼ������ζ������Ҫֱ���޸�����Ķ�ά�����벻Ҫ ʹ����һ����������תͼ��

ʾ�� 1��
���룺matrix = [[1,2,3],[4,5,6],[7,8,9]]
�����[[7,4,1],[8,5,2],[9,6,3]]

ʾ�� 2��
���룺matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
�����[[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]

��ʾ��
n == matrix.length == matrix[i].length
1 <= n <= 20
-1000 <= matrix[i][j] <= 1000

��ע��
�ٷ�ʾ������ת������ת�����ȴ�ֱ��ת�����Խ��߷�ת��
*/
void Top100Liked::Solution::rotate(vector<vector<int>>& matrix, bool isMyOwn)
{
	if (isMyOwn) {
		int num = matrix.size();
		int circle = num / 2;
		num--;
		bool doubleFlag = (num % 2 == 0);
		// ��ʽѭ������
		for (int i = 0; i < circle; i++)
		{
			int temp = 0;
			for (int j = 0; j < circle; j++)
			{
				temp = matrix[j][num - i];
				matrix[j][num - i] = matrix[i][j];
				matrix[i][j] = temp;

				temp = matrix[num - i][num - j];
				matrix[num - i][num - j] = matrix[i][j];
				matrix[i][j] = temp;

				temp = matrix[num - j][i];
				matrix[num - j][i] = matrix[i][j];
				matrix[i][j] = temp;
			}
			if (doubleFlag) {
				temp = matrix[circle][num - i];
				matrix[circle][num - i] = matrix[i][circle];
				matrix[i][circle] = temp;

				temp = matrix[num - i][num - circle];
				matrix[num - i][num - circle] = matrix[i][circle];
				matrix[i][circle] = temp;

				temp = matrix[num - circle][i];
				matrix[num - circle][i] = matrix[i][circle];
				matrix[i][circle] = temp;
			}
		}
	}
	else {
		int n = matrix.size();
		// ˮƽ��ת
		for (int i = 0; i < n / 2; ++i) {
			for (int j = 0; j < n; ++j) {
				swap(matrix[i][j], matrix[n - i - 1][j]);
			}
		}
		// ���Խ��߷�ת
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < i; ++j) {
				swap(matrix[i][j], matrix[j][i]);
			}
		}
	}
}

/*
��дһ����Ч���㷨������ m x n ���� matrix �е�һ��Ŀ��ֵ target ���þ�������������ԣ�
ÿ�е�Ԫ�ش������������С�
ÿ�е�Ԫ�ش��ϵ����������С�

ʾ�� 1��
���룺matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 5
�����true

ʾ�� 2��
���룺matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 20
�����false

��ʾ��
m == matrix.length
n == matrix[i].length
1 <= n, m <= 300
-10��9�η� <= matrix[i][j] <= 10��9�η�
ÿ�е�����Ԫ�ش�������������
ÿ�е�����Ԫ�ش��ϵ�����������
-10��9�η� <= target <= 10��9�η�

��ע��
�ٷ�����һ��ʼ���Ǵӵ�һ�����Ŀ�ʼ�Ƚϣ������Ǵ� 0��0 ��ʼ��
���ڶ��Ǵ�С�����˳�򣬹ʵ�left--�󣬲�����Ҫ��������һ������left++���������Ϊ[top][left]����target����[top+1][left]Ҳ�ض�����target
*/
bool Top100Liked::Solution::searchMatrix(vector<vector<int>>& matrix, int target, bool isMyOwn)
{
	if (isMyOwn) {
		int m = matrix.size();
		int n = matrix[0].size();
		int left = 0, top = 0;
		while (left < n && top < m) {
			if (matrix[top][left] == target) {
				return true;
			}
			else if (matrix[top][left] < target && left < n - 1) {
				left++;
			}
			else {
				top++;
				while (top < m && left > 0 && matrix[top][left] > target) {
					left--;
				}
			}
		}
		return false;
	}
	else {
		int m = matrix.size(), n = matrix[0].size();
		int x = 0, y = n - 1;
		while (x < m && y >= 0) {
			if (matrix[x][y] == target) {
				return true;
			}
			if (matrix[x][y] > target) {
				--y;
			}
			else {
				++x;
			}
		}
		return false;
	}
}

/*
���������������ͷ�ڵ� headA �� headB �������ҳ������������������ཻ����ʼ�ڵ㡣����������������ཻ�ڵ㣬���� null ��

ͼʾ���������ڽڵ� c1 ��ʼ�ཻ��
A:    a1-a2-c1-c2-c3
B: b1-b2-b3-c1-c2-c3

��Ŀ���� ��֤ ������ʽ�ṹ�в����ڻ���
ע�⣬�������ؽ����������� ������ԭʼ�ṹ ��
�Զ������⣺
����ϵͳ ���������£�����Ƶĳ��� ������ �����룩��
intersectVal - �ཻ����ʼ�ڵ��ֵ������������ཻ�ڵ㣬��һֵΪ 0
listA - ��һ������
listB - �ڶ�������
skipA - �� listA �У���ͷ�ڵ㿪ʼ����������ڵ�Ľڵ���
skipB - �� listB �У���ͷ�ڵ㿪ʼ����������ڵ�Ľڵ���
����ϵͳ��������Щ���봴����ʽ���ݽṹ����������ͷ�ڵ� headA �� headB ���ݸ���ĳ�����������ܹ���ȷ�����ཻ�ڵ㣬��ô��Ľ���������� ������ȷ�� ��

ʾ�� 1��
A:   4-1-8-4-5
B: 5-6-1-8-4-5
���룺intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5], skipA = 2, skipB = 3
�����Intersected at '8'
���ͣ��ཻ�ڵ��ֵΪ 8 ��ע�⣬������������ཻ����Ϊ 0����
�Ӹ��Եı�ͷ��ʼ�������� A Ϊ [4,1,8,4,5]������ B Ϊ [5,6,1,8,4,5]��
�� A �У��ཻ�ڵ�ǰ�� 2 ���ڵ㣻�� B �У��ཻ�ڵ�ǰ�� 3 ���ڵ㡣
�� ��ע���ཻ�ڵ��ֵ��Ϊ 1����Ϊ������ A ������ B ֮��ֵΪ 1 �Ľڵ� (A �еڶ����ڵ�� B �е������ڵ�) �ǲ�ͬ�Ľڵ㡣���仰˵���������ڴ���ָ��������ͬ��λ�ã������� A ������ B ��ֵΪ 8 �Ľڵ� (A �е������ڵ㣬B �е��ĸ��ڵ�) ���ڴ���ָ����ͬ��λ�á�


ʾ�� 2��
A: 1-9-1-2-4
B:     3-2-4
���룺intersectVal = 2, listA = [1,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
�����Intersected at '2'
���ͣ��ཻ�ڵ��ֵΪ 2 ��ע�⣬������������ཻ����Ϊ 0����
�Ӹ��Եı�ͷ��ʼ�������� A Ϊ [1,9,1,2,4]������ B Ϊ [3,2,4]��
�� A �У��ཻ�ڵ�ǰ�� 3 ���ڵ㣻�� B �У��ཻ�ڵ�ǰ�� 1 ���ڵ㡣

ʾ�� 3��
A: 2-6-4
B: 1-5
���룺intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
�����No intersection
���ͣ��Ӹ��Եı�ͷ��ʼ�������� A Ϊ [2,6,4]������ B Ϊ [1,5]��
���������������ཻ������ intersectVal ����Ϊ 0���� skipA �� skipB ����������ֵ��
�����������ཻ����˷��� null ��

��ʾ��
listA �нڵ���ĿΪ m
listB �нڵ���ĿΪ n
1 <= m, n <= 3 * 10��4�η�
1 <= Node.val <= 10��5�η�
0 <= skipA <= m
0 <= skipB <= n
��� listA �� listB û�н��㣬intersectVal Ϊ 0
��� listA �� listB �н��㣬intersectVal == listA[skipA] == listB[skipB]

���ף����ܷ����һ��ʱ�临�Ӷ� O(m + n) ������ O(1) �ڴ�Ľ��������

��ע��
������ֱ�ӱȽ�ֵ�����ǱȽ� �ڴ��еĵ�ַ
*/
Top100Liked::Solution::ListNode* Top100Liked::Solution::getIntersectionNode(ListNode* headA, ListNode* headB, bool isMyOwn)
{
	isMyOwn = false;
	if (isMyOwn) {
		vector<int>tempA;
		vector<int>tempB;
		Top100Liked::Solution::ListNode* retNode = headB;
		while (headA != nullptr) {
			tempA.push_back(headA->val);
			headA = headA->next;
		}
		while (headB != nullptr) {
			tempB.push_back(headB->val);
			headB = headB->next;
		}
		int index = 0;
		int m = tempA.size();
		int n = tempB.size();
		for (int i = n - 1; i > 0; i--) {
			if (tempB[i] != tempA[m - n + i]) {
				break;
			}
			index++;
		}
		if (index != 0) {
			for (int i = 0; i < n - index; i++) {
				retNode = retNode->next;
			}
			return retNode;
		}
		return nullptr;
	}
	else {
		if (headA == nullptr || headB == nullptr) {
			return nullptr;
		}
		ListNode* pA = headA, * pB = headB;
		while (pA != pB) {
			pA = pA == nullptr ? headB : pA->next;
			pB = pB == nullptr ? headA : pB->next;
		}
		return pA;
	}
}

/*
���㵥�����ͷ�ڵ� head �����㷴ת���������ط�ת�������

ʾ�� 1��
���룺head = [1,2,3,4,5]
�����[5,4,3,2,1]

ʾ�� 2��
���룺head = [1,2]
�����[2,1]

ʾ�� 3��
���룺head = []
�����[]

��ʾ��
�����нڵ����Ŀ��Χ�� [0, 5000]
-5000 <= Node.val <= 5000

���ף��������ѡ�õ�����ݹ鷽ʽ��ɷ�ת�����ܷ������ַ����������⣿

��ע��
�ٷ�������û�����´����½ڵ��ڵݹ���ӻ��������ǵݹ��βָ��ͷ�����ӣ���ͬʱ��ϴ�ͷָ��β������
*/
Top100Liked::Solution::ListNode2* Top100Liked::Solution::reverseList(ListNode2* head, bool isMyOwn)
{
	if (isMyOwn) {
		if (head != NULL && head->next != NULL)
		{
			ListNode2* p = new ListNode2;
			int val = head->val;
			p->val = val;
			head = reverseList(head->next, isMyOwn);
			ListNode2* cur = head;
			while (head->next != NULL)
			{
				head = head->next;
			}
			head->next = p;
			head = cur;
		}
		return head;
	}
	else {
		if (!head || !head->next) {
			return head;
		}
		ListNode2* newHead = reverseList(head->next, isMyOwn);
		head->next->next = head;
		head->next = nullptr;
		return newHead;
	}
}

void backtrack(vector<vector<int>>& res, vector<int>& output, int first, int len) {
	// ��������������
	if (first == len) {
		res.emplace_back(output);
		return;
	}
	for (int i = first; i < len; ++i) {
		// ��̬ά������
		swap(output[i], output[first]);
		// �����ݹ�����һ����
		backtrack(res, output, first + 1, len);
		// ��������
		swap(output[i], output[first]);
	}
}
/*
����һ�������ظ����ֵ����� nums �������� ���п��ܵ�ȫ���� ������� ������˳�� ���ش𰸡�

ʾ�� 1��
���룺nums = [1,2,3]
�����[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

ʾ�� 2��
���룺nums = [0,1]
�����[[0,1],[1,0]]

ʾ�� 3��
���룺nums = [1]
�����[[1]]

��ʾ��
1 <= nums.length <= 6
-10 <= nums[i] <= 10
nums �е��������� ������ͬ

��ע��
�ٷ�������дһ���ݹ麯��������һ�ν���ѭ��ʱ��fitst��n�ֿ��ܣ�������n�ε���һλ������ݹ��ͬ��ڶ�λ��n-1�ֿ��ܣ���ɵݹ���轫���ݸ�ԭ������Ӱ������������ݵ����
*/
vector<vector<int>> Top100Liked::Solution::permute(vector<int>& nums, bool isMyOwn)
{
	if (isMyOwn) {
		vector<vector<int>>ret;
		if (nums.size() == 1) {
			vector<int> start{ nums[0] };
			ret.push_back(start);
		}
		else {
			for (int i = 0; i < nums.size(); i++) {
				vector<int> newVec = nums;
				newVec.erase(find(newVec.begin(), newVec.end(), nums[i]));
				auto temp = permute(newVec, isMyOwn);
				for (auto item : temp) {
					if (find(item.begin(), item.end(), nums[i]) == item.end()) {
						item.push_back(nums[i]);
						ret.push_back(item);
					}
				}
			}
		}
		return ret;
	}
	else {
		vector<vector<int>> res;
		backtrack(res, nums, 0, (int)nums.size());
		return res;
	}
}

void myLoop(vector<string>& res, string output, int l, int r, int n) {
	// ������С��n��������������������
	if (r == 0 && l == 0) {
		res.push_back(output);
	}
	// �ȷ�������
	if (l > 0) {
		myLoop(res, output + "(", l - 1, r, n);
	}
	// ��������
	if (r > l) {
		myLoop(res, output + ")", l, r - 1, n);
	}
}
void backtrack(vector<string>& ans, string& cur, int open, int close, int n) {
	if (cur.size() == n * 2) {
		ans.push_back(cur);
		return;
	}
	if (open < n) {
		cur.push_back('(');
		backtrack(ans, cur, open + 1, close, n);
		cur.pop_back();
	}
	if (close < open) {
		cur.push_back(')');
		backtrack(ans, cur, open, close + 1, n);
		cur.pop_back();
	}
}
/*
���� n �����������ŵĶ������������һ�������������ܹ��������п��ܵĲ��� ��Ч�� ������ϡ�

ʾ�� 1��
���룺n = 3
�����["((()))","(()())","(())()","()(())","()()()"]

ʾ�� 2��
���룺n = 1
�����["()"]

��ʾ��
1 <= n <= 8

��ע��
�ٷ�����˼·һ�£�ֻ�Ǵ�n��ʼ������˴�0��ʼ���������ݲ�����
*/
vector<string> Top100Liked::Solution::generateParenthesis(int n, bool isMyOwn)
{
	if (isMyOwn) {
		vector<string> ret;
		myLoop(ret, "", n, n, n);
		return ret;
	}
	else {
		vector<string> result;
		string current;
		backtrack(result, current, 0, 0, n);
		return result;
	}
}





