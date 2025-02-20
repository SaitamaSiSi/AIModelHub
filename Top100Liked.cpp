#include "Top100Liked.h"

/*
给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。
你可以假设每种输入只会对应一个答案，并且你不能使用两次相同的元素。
你可以按任意顺序返回答案。

示例 1：
输入：nums = [2,7,11,15], target = 9
输出：[0,1]
解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。

示例 2：
输入：nums = [3,2,4], target = 6
输出：[1,2]

示例 3：
输入：nums = [3,3], target = 6
输出：[0,1]

提示：
2 <= nums.length <= 104
-109 <= nums[i] <= 109
-109 <= target <= 109
只会存在一个有效答案

进阶：你可以想出一个时间复杂度小于 O(n2) 的算法吗？
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
给你一个字符串数组，请你将 字母异位词 组合在一起。可以按任意顺序返回结果列表。
字母异位词 是由重新排列源单词的所有字母得到的一个新单词。

示例 1:
输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
输出: [["bat"],["nat","tan"],["ate","eat","tea"]]

示例 2:
输入: strs = [""]
输出: [[""]]

示例 3:
输入: strs = ["a"]
输出: [["a"]]

提示：
1 <= strs.length <= 104
0 <= strs[i].length <= 100
strs[i] 仅包含小写字母
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
给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。
请你设计并实现时间复杂度为 O(n) 的算法解决此问题。

示例 1：
输入：nums = [100,4,200,1,3,2]
输出：4
解释：最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4。

示例 2：
输入：nums = [0,3,7,2,5,8,4,6,0,1]
输出：9

示例 3：
输入：nums = [1,0,1,2]
输出：3

提示：
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
给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
请注意 ，必须在不复制数组的情况下原地对数组进行操作。

示例 1:
输入: nums = [0,1,0,3,12]
输出: [1,3,12,0,0]

示例 2:
输入: nums = [0]
输出: [0]

提示:
1 <= nums.length <= 104
-231 <= nums[i] <= 231 - 1

进阶：你能尽量减少完成的操作次数吗？
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
给定一个长度为 n 的整数数组 height 。有 n 条垂线，第 i 条线的两个端点是 (i, 0) 和 (i, height[i]) 。
找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
返回容器可以储存的最大水量。
说明：你不能倾斜容器。

示例 1：
输入：[1,8,6,2,5,4,8,3,7]
输出：49
解释：图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49。

示例 2：
输入：height = [1,1]
输出：1

提示：
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
给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ，同时还满足 nums[i] + nums[j] + nums[k] == 0 。请你返回所有和为 0 且不重复的三元组。
注意：答案中不可以包含重复的三元组。

示例 1：
输入：nums = [-1,0,1,2,-1,-4]
输出：[[-1,-1,2],[-1,0,1]]
解释：
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0 。
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0 。
nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0 。
不同的三元组是 [-1,0,1] 和 [-1,-1,2] 。
注意，输出的顺序和三元组的顺序并不重要。

示例 2：
输入：nums = [0,1,1]
输出：[]
解释：唯一可能的三元组和不为 0 。

示例 3：
输入：nums = [0,0,0]
输出：[[0,0,0]]
解释：唯一可能的三元组和为 0 。

提示：
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

		//循环最左边第一个数
		for (int first = 0; first < n; ++first)
		{
			// 使得下一次的数和上一次不同
			if (first > 0 && nums[first] == nums[first - 1])
			{
				continue;
			}
			// 将第三个数放在最右边
			int third = n - 1;
			// a+b+c=0 转换成 b+c=-a;
			int target = -nums[first];
			// 循环左边第二个数，在第一个数后面
			for (int second = first + 1; second < n; ++second)
			{
				// 使得下一次的数和上一次不同
				if (second > first + 1 && nums[second] == nums[second - 1])
				{
					continue;
				}
				// 让第三个数一直在第二个数右边
				while (second < third && nums[second] + nums[third] > target)
				{
					--third;
				}
				//当重合后就遍历完了b，c 退出第一次循环
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
		// 枚举 a
		for (int first = 0; first < n; ++first) {
			// 需要和上一次枚举的数不相同
			if (first > 0 && nums[first] == nums[first - 1]) {
				continue;
			}
			// c 对应的指针初始指向数组的最右端
			int third = n - 1;
			int target = -nums[first];
			// 枚举 b
			for (int second = first + 1; second < n; ++second) {
				// 需要和上一次枚举的数不相同
				if (second > first + 1 && nums[second] == nums[second - 1]) {
					continue;
				}
				// 需要保证 b 的指针在 c 的指针的左侧
				while (second < third && nums[second] + nums[third] > target) {
					--third;
				}
				// 如果指针重合，随着 b 后续的增加
				// 就不会有满足 a+b+c=0 并且 b<c 的 c 了，可以退出循环
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
给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

示例 1：
输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。

示例 2：
输入：height = [4,2,0,3,2,5]
输出：9

提示：
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
给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串的长度。

示例 1:
输入: s = "abcabcbb"
输出: 3
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。

示例 2:
输入: s = "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。

示例 3:
输入: s = "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
	 请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。

提示：
0 <= s.length <= 5 * 104
s 由英文字母、数字、符号和空格组成
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
		// 哈希集合，记录每个字符是否出现过
		unordered_set<char> occ;
		int n = s.size();
		// 右指针，初始值为 -1，相当于我们在字符串的左边界的左侧，还没有开始移动
		int rk = -1, ans = 0;
		// 枚举左指针的位置，初始值隐性地表示为 -1
		for (int i = 0; i < n; ++i) {
			if (i != 0) {
				// 左指针向右移动一格，移除一个字符
				occ.erase(s[i - 1]);
			}
			while (rk + 1 < n && !occ.count(s[rk + 1])) {
				// 不断地移动右指针
				occ.insert(s[rk + 1]);
				++rk;
			}
			// 第 i 到 rk 个字符是一个极长的无重复字符子串
			ans = max(ans, rk - i + 1);
		}
		return ans;
	}
}

/*
给定两个字符串 s 和 p，找到 s 中所有 p 的异位词的子串，返回这些子串的起始索引。不考虑答案输出的顺序。

示例 1:
输入: s = "cbaebabacd", p = "abc"
输出: [0,6]
解释:
起始索引等于 0 的子串是 "cba", 它是 "abc" 的异位词。
起始索引等于 6 的子串是 "bac", 它是 "abc" 的异位词。

示例 2:
输入: s = "abab", p = "ab"
输出: [0,1,2]
解释:
起始索引等于 0 的子串是 "ab", 它是 "ab" 的异位词。
起始索引等于 1 的子串是 "ba", 它是 "ab" 的异位词。
起始索引等于 2 的子串是 "ab", 它是 "ab" 的异位词。

提示:
1 <= s.length, p.length <= 3 * 104
s 和 p 仅包含小写字母

备注:
differ表示有多少个字母不同，只要count中存在一个不为0的字母，则就是有一个不同，窗口长度一直为sLen-pLen，如果左侧待移除字母的原计数为1，则计数减1则为0，即不同减1，同理，右侧待加入字母为-1，则加入后为0，不同减1，同理differ加一也是类似原理。
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
			if (count[s[i] - 'a'] == 1) {  // 窗口中字母 s[i] 的数量与字符串 p 中的数量从不同变得相同
				--differ;
			}
			else if (count[s[i] - 'a'] == 0) {  // 窗口中字母 s[i] 的数量与字符串 p 中的数量从相同变得不同
				++differ;
			}
			--count[s[i] - 'a'];

			if (count[s[i + pLen] - 'a'] == -1) {  // 窗口中字母 s[i+pLen] 的数量与字符串 p 中的数量从不同变得相同
				--differ;
			}
			else if (count[s[i + pLen] - 'a'] == 0) {  // 窗口中字母 s[i+pLen] 的数量与字符串 p 中的数量从相同变得不同
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
给你一个整数数组 nums 和一个整数 k ，请你统计并返回 该数组中和为 k 的子数组的个数 。
子数组是数组中元素的连续非空序列。

示例 1：
输入：nums = [1,1,1], k = 2
输出：2

示例 2：
输入：nums = [1,2,3], k = 3
输出：2

提示：
1 <= nums.length <= 2 * 104
-1000 <= nums[i] <= 1000
-107 <= k <= 107

备注:
其中pre为前面累计总和，通过sum[i-j]的位置中间和为k的话，那么sum[0-j]-sum[0,i]就是sum[i-j]，也就是k值，然后从0-n的所有累计值都出现过，所以只要差值为k，且前面的sum[0-i]出现过即可。
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
给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。
返回 滑动窗口中的最大值 。

示例 1：
输入：nums = [1,3,-1,-3,5,3,6,7], k = 3
输出：[3,3,5,5,6,7]
解释：
滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7

示例 2：
输入：nums = [1], k = 1
输出：[1]

示例 3：
输入：nums = [1,3,1,2,0,5], k = 3
输出：[3,3,2,5]

示例 4：
输入：nums = [-7,-8,7,5,7,1,6,0], k = 4
输出：[7,7,7,7,7]

提示：
1 <= nums.length <= 105
-104 <= nums[i] <= 104
1 <= k <= nums.length

备注：
其原理是根据k来分段，先从左往右，每组内持续比较，设置最大值，即从小到大，为p，同理从右往左，持续比较设置最大值，也是从小到大，为s。
当刚好在k大小内时，则p最右侧和s最左侧都是最大值，当不为k的倍数时，则横跨两个k长度分组，那么就需要比较p范围内最右侧的最大值和s范围内最左侧的最大值，得到最大值。
同时需要考虑长度不是刚好为k的倍数的情况，即初始值[0]的[n-1]的值，在示例方法中增加了该情况的考虑。
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
给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。

注意：
对于 t 中重复字符，我们寻找的子字符串中该字符数量必须不少于 t 中该字符数量。
如果 s 中存在这样的子串，我们保证它是唯一的答案。

示例 1：
输入：s = "ADOBECODEBANC", t = "ABC"
输出："BANC"
解释：最小覆盖子串 "BANC" 包含来自字符串 t 的 'A'、'B' 和 'C'。

示例 2：
输入：s = "a", t = "a"
输出："a"
解释：整个字符串 s 是最小覆盖子串。

示例 3:
输入: s = "a", t = "aa"
输出: ""
解释: t 中两个字符 'a' 均应包含在 s 的子串中，
因此没有符合条件的子字符串，返回空字符串。

提示：
m == s.length
n == t.length
1 <= m, n <= 105
s 和 t 由英文字母组成

进阶：你能设计一个在 o(m+n) 时间内解决此问题的算法吗？
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


