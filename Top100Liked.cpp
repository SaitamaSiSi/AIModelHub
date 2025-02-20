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
-231 <= nums[i] <= 231 - 1

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


