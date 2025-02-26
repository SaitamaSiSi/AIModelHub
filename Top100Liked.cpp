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
-2的31次方 <= nums[i] <= 2的31次方 - 1

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
给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
子数组是数组中的一个连续部分。

示例 1：
输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。

示例 2：
输入：nums = [1]
输出：1

示例 3：
输入：nums = [5,4,-1,7,8]
输出：23

提示：
1 <= nums.length <= 105
-104 <= nums[i] <= 104

进阶：如果你已经实现复杂度为 O(n) 的解法，尝试使用更为精妙的 分治法 求解。
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
以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。

示例 1：
输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
输出：[[1,6],[8,10],[15,18]]
解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].

示例 2：
输入：intervals = [[1,4],[4,5]]
输出：[[1,5]]
解释：区间 [1,4] 和 [4,5] 可被视为重叠区间。

示例 3：
输入：intervals = [[2,5],[1,5]]
输出：[[1,5]]

示例 4：
输入：intervals = [[1,5],[2,3]]
输出：[[1,5]]

提示：
1 <= intervals.length <= 104
intervals[i].length == 2
0 <= starti <= endi <= 104

备注：
数据已经经过排序，大小方面的比较相对简单处理，仅仅需要考虑新数据左侧是否大于尾部数据右侧即可。
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
给定一个整数数组 nums，将数组中的元素向右轮转 k 个位置，其中 k 是非负数。

示例 1:
输入: nums = [1,2,3,4,5,6,7], k = 3
输出: [5,6,7,1,2,3,4]
解释:
向右轮转 1 步: [7,1,2,3,4,5,6]
向右轮转 2 步: [6,7,1,2,3,4,5]
向右轮转 3 步: [5,6,7,1,2,3,4]

示例 2:
输入：nums = [-1,-100,3,99], k = 2
输出：[3,99,-1,-100]
解释:
向右轮转 1 步: [99,-1,-100,3]
向右轮转 2 步: [3,99,-1,-100]

提示：
1 <= nums.length <= 105
-2的31次方 <= nums[i] <= 2的31次方 - 1
0 <= k <= 105

进阶：
尽可能想出更多的解决方案，至少有 三种 不同的方法可以解决这个问题。
你可以使用空间复杂度为 O(1) 的 原地 算法解决这个问题吗？

备注：
因为是按照顺序从右侧移出，并添加到左侧，先出先进，相当于翻转了一次。
所以可以先进行翻转，再由于先出先进，交换k步长的顺序，完成移出部分数据的顺序和位置，再将剩余数据翻转回原顺序和位置即可。
*/
void Top100Liked::Solution::rotate(vector<int>& nums, int k, bool isMyOwn)
{
	if (isMyOwn) {
		if (nums.size() <= 1) { return; }
		int sweap = 0;
		// 1,2,3,4,5,6,7
		int num = nums.size();
		// 除去多余的循环移动步数
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

		// 环状替换，暂不推荐
		//int n = nums.size();
		//k = k % n;
		//// 计算最大公约数, (48, 18) => 6, (-24, 0) => 24
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
给你一个整数数组 nums，返回 数组 answer ，其中 answer[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积 。
题目数据 保证 数组 nums之中任意元素的全部前缀元素和后缀的乘积都在  32 位 整数范围内。
请 不要使用除法，且在 O(n) 时间复杂度内完成此题。

示例 1:
输入: nums = [1,2,3,4]
输出: [24,12,8,6]

示例 2:
输入: nums = [-1,1,0,-3,3]
输出: [0,0,9,0,0]

提示：
2 <= nums.length <= 105
-30 <= nums[i] <= 30
输入 保证 数组 answer[i] 在  32 位 整数范围内

进阶：你可以在 O(1) 的额外空间复杂度内完成这个题目吗？（ 出于对空间复杂度分析的目的，输出数组 不被视为 额外空间。）

备注：
时间复杂度 O(2n) 可以近似看成 O(n)，所以循环常量次数也可以符合题意
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

		// answer[i] 表示索引 i 左侧所有元素的乘积
		// 因为索引为 '0' 的元素左侧没有元素， 所以 answer[0] = 1
		answer[0] = 1;
		for (int i = 1; i < length; i++) {
			answer[i] = nums[i - 1] * answer[i - 1];
		}

		// R 为右侧所有元素的乘积
		// 刚开始右边没有元素，所以 R = 1
		int R = 1;
		for (int i = length - 1; i >= 0; i--) {
			// 对于索引 i，左边的乘积为 answer[i]，右边的乘积为 R
			answer[i] = answer[i] * R;
			// R 需要包含右边所有的乘积，所以计算下一个结果时需要将当前值乘到 R 上
			R *= nums[i];
		}
		return answer;
	}
}

/*
给你一个未排序的整数数组 nums ，请你找出其中没有出现的最小的正整数。
请你实现时间复杂度为 O(n) 并且只使用常数级别额外空间的解决方案。

示例 1：
输入：nums = [1,2,0]
输出：3
解释：范围 [1,2] 中的数字都在数组中。

示例 2：
输入：nums = [3,4,-1,1]
输出：2
解释：1 在数组中，但 2 没有。

示例 3：
输入：nums = [7,8,9,11,12]
输出：1
解释：最小的正数 1 没有出现。

提示：
1 <= nums.length <= 105
-2的31次方 <= nums[i] <= 2的31次方 - 1

备注：
官方示例方式是采用哈希表的逻辑实现，哈希表本身也是一个数组，将i放在角标i-1的位置。
比如 3 4 -1 1
将3放在角标2上后，-1 4 3 1 ，还需要再次检测-1，不符合条件，不用再调整
将4放在角标3上后，-1 1 3 4 ，还需再次检测下1，符合条件，且不在自身位置，再次调整
将1放在角标0上后，1 -1 3 4 ，还需再次检测下-1，不符合条件，不用再调整
后续3 4 都在自身位置，无需再调整
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
给定一个 m x n 的矩阵，如果一个元素为 0 ，则将其所在行和列的所有元素都设为 0 。请使用 原地 算法。

示例 1：
输入：matrix = [[1,1,1],[1,0,1],[1,1,1]]
输出：[[1,0,1],[0,0,0],[1,0,1]]

示例 2：
输入：matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
输出：[[0,0,0,0],[0,4,5,0],[0,3,1,0]]

提示：
m == matrix.length
n == matrix[0].length
1 <= m, n <= 200
-2的31次方 <= matrix[i][j] <= 2的31次方 - 1

进阶：
一个直观的解决方案是使用  O(mn) 的额外空间，但这并不是一个好的解决方案。
一个简单的改进方案是使用 O(m + n) 的额外空间，但这仍然不是最好的解决方案。
你能想出一个仅使用常量空间的解决方案吗？
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
给你一个 m 行 n 列的矩阵 matrix ，请按照 顺时针螺旋顺序 ，返回矩阵中的所有元素。

示例 1：
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,3,6,9,8,7,4,5]

示例 2：
输入：matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
输出：[1,2,3,4,8,12,11,10,9,5,6,7]

提示：
m == matrix.length
n == matrix[i].length
1 <= m, n <= 10
-100 <= matrix[i][j] <= 100

备注：
将螺旋矩阵看作四面墙，每次移动后，墙壁都往内缩小。
*/
vector<int> Top100Liked::Solution::spiralOrder(vector<vector<int>>& matrix, bool isMyOwn)
{
	if (isMyOwn) {
		int m = matrix.size();
		int n = matrix[0].size();
		vector<int> result;
		int start = 0, top = 0, left = 0, right = n - 1, bottom = m - 1;
		while (true) {
			// 层级下移
			for (int i = left; i <= right; i++) {
				result.push_back(matrix[top][i]);
				start++;
			}
			top++;
			if (start >= m * n) { break; }

			// 层级左移
			for (int i = top; i <= bottom; i++) {
				result.push_back(matrix[i][right]);
				start++;
			}
			right--;
			if (start >= m * n) { break; }

			// 层级上移
			for (int i = right; i >= left; i--) {
				result.push_back(matrix[bottom][i]);
				start++;
			}
			bottom--;
			if (start >= m * n) { break; }

			// 层级右移
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
给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。
你必须在 原地 旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像。

示例 1：
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[[7,4,1],[8,5,2],[9,6,3]]

示例 2：
输入：matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
输出：[[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]

提示：
n == matrix.length == matrix[i].length
1 <= n <= 20
-1000 <= matrix[i][j] <= 1000

备注：
官方示例将旋转看作翻转处理，先垂直翻转再主对角线翻转。
*/
void Top100Liked::Solution::rotate(vector<vector<int>>& matrix, bool isMyOwn)
{
	if (isMyOwn) {
		int num = matrix.size();
		int circle = num / 2;
		num--;
		bool doubleFlag = (num % 2 == 0);
		// 链式循环交换
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
		// 水平翻转
		for (int i = 0; i < n / 2; ++i) {
			for (int j = 0; j < n; ++j) {
				swap(matrix[i][j], matrix[n - i - 1][j]);
			}
		}
		// 主对角线翻转
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < i; ++j) {
				swap(matrix[i][j], matrix[j][i]);
			}
		}
	}
}

/*
编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：
每行的元素从左到右升序排列。
每列的元素从上到下升序排列。

示例 1：
输入：matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 5
输出：true

示例 2：
输入：matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 20
输出：false

提示：
m == matrix.length
n == matrix[i].length
1 <= n, m <= 300
-10的9次方 <= matrix[i][j] <= 10的9次方
每行的所有元素从左到右升序排列
每列的所有元素从上到下升序排列
-10的9次方 <= target <= 10的9次方

备注：
官方答案是一开始就是从第一行最大的开始比较，并不是从 0，0 开始。
由于都是从小到大的顺序，故当left--后，并不需要考虑在下一层会出现left++的情况，因为[top][left]大于target，故[top+1][left]也必定大于target
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
给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 null 。

图示两个链表在节点 c1 开始相交：
A:    a1-a2-c1-c2-c3
B: b1-b2-b3-c1-c2-c3

题目数据 保证 整个链式结构中不存在环。
注意，函数返回结果后，链表必须 保持其原始结构 。
自定义评测：
评测系统 的输入如下（你设计的程序 不适用 此输入）：
intersectVal - 相交的起始节点的值。如果不存在相交节点，这一值为 0
listA - 第一个链表
listB - 第二个链表
skipA - 在 listA 中（从头节点开始）跳到交叉节点的节点数
skipB - 在 listB 中（从头节点开始）跳到交叉节点的节点数
评测系统将根据这些输入创建链式数据结构，并将两个头节点 headA 和 headB 传递给你的程序。如果程序能够正确返回相交节点，那么你的解决方案将被 视作正确答案 。

示例 1：
A:   4-1-8-4-5
B: 5-6-1-8-4-5
输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5], skipA = 2, skipB = 3
输出：Intersected at '8'
解释：相交节点的值为 8 （注意，如果两个链表相交则不能为 0）。
从各自的表头开始算起，链表 A 为 [4,1,8,4,5]，链表 B 为 [5,6,1,8,4,5]。
在 A 中，相交节点前有 2 个节点；在 B 中，相交节点前有 3 个节点。
— 请注意相交节点的值不为 1，因为在链表 A 和链表 B 之中值为 1 的节点 (A 中第二个节点和 B 中第三个节点) 是不同的节点。换句话说，它们在内存中指向两个不同的位置，而链表 A 和链表 B 中值为 8 的节点 (A 中第三个节点，B 中第四个节点) 在内存中指向相同的位置。


示例 2：
A: 1-9-1-2-4
B:     3-2-4
输入：intersectVal = 2, listA = [1,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
输出：Intersected at '2'
解释：相交节点的值为 2 （注意，如果两个链表相交则不能为 0）。
从各自的表头开始算起，链表 A 为 [1,9,1,2,4]，链表 B 为 [3,2,4]。
在 A 中，相交节点前有 3 个节点；在 B 中，相交节点前有 1 个节点。

示例 3：
A: 2-6-4
B: 1-5
输入：intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
输出：No intersection
解释：从各自的表头开始算起，链表 A 为 [2,6,4]，链表 B 为 [1,5]。
由于这两个链表不相交，所以 intersectVal 必须为 0，而 skipA 和 skipB 可以是任意值。
这两个链表不相交，因此返回 null 。

提示：
listA 中节点数目为 m
listB 中节点数目为 n
1 <= m, n <= 3 * 10的4次方
1 <= Node.val <= 10的5次方
0 <= skipA <= m
0 <= skipB <= n
如果 listA 和 listB 没有交点，intersectVal 为 0
如果 listA 和 listB 有交点，intersectVal == listA[skipA] == listB[skipB]

进阶：你能否设计一个时间复杂度 O(m + n) 、仅用 O(1) 内存的解决方案？

备注：
并不是直接比较值，而是比较 内存中的地址
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
给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。

示例 1：
输入：head = [1,2,3,4,5]
输出：[5,4,3,2,1]

示例 2：
输入：head = [1,2]
输出：[2,1]

示例 3：
输入：head = []
输出：[]

提示：
链表中节点的数目范围是 [0, 5000]
-5000 <= Node.val <= 5000

进阶：链表可以选用迭代或递归方式完成反转。你能否用两种方法解决这道题？

备注：
官方方法并没有重新创建新节点在递归添加回来，而是递归从尾指向开头的链接，并同时打断从头指向尾的链接
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
	// 所有数都填完了
	if (first == len) {
		res.emplace_back(output);
		return;
	}
	for (int i = first; i < len; ++i) {
		// 动态维护数组
		swap(output[i], output[first]);
		// 继续递归填下一个数
		backtrack(res, output, first + 1, len);
		// 撤销操作
		swap(output[i], output[first]);
	}
}
/*
给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。

示例 1：
输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

示例 2：
输入：nums = [0,1]
输出：[[0,1],[1,0]]

示例 3：
输入：nums = [1]
输出：[[1]]

提示：
1 <= nums.length <= 6
-10 <= nums[i] <= 10
nums 中的所有整数 互不相同

备注：
官方采用另写一个递归函数，当第一次进入循环时，fitst有n种可能，即交换n次到第一位，进入递归后，同理第二位有n-1种可能，完成递归后，需将数据复原，避免影响放置其他数据的情况
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

void myLoop(vector<string>& res, string output, int l, int r) {
	// 左括号小于n并且右括号少于左括号
	if (r == 0 && l == 0) {
		res.push_back(output);
	}
	// 先放左括号
	if (l > 0) {
		myLoop(res, output + "(", l - 1, r);
	}
	// 后补右括号
	if (r > l) {
		myLoop(res, output + ")", l, r - 1);
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
数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。

示例 1：
输入：n = 3
输出：["((()))","(()())","(())()","()(())","()()()"]

示例 2：
输入：n = 1
输出：["()"]

提示：
1 <= n <= 8

备注：
官方大致思路一致，只是从n开始，变成了从0开始，方法三暂不考虑
*/
vector<string> Top100Liked::Solution::generateParenthesis(int n, bool isMyOwn)
{
	if (isMyOwn) {
		vector<string> ret;
		myLoop(ret, "", n, n);
		return ret;
	}
	else {
		vector<string> result;
		string current;
		backtrack(result, current, 0, 0, n);
		return result;
	}
}

vector<string> myLoop2(vector<vector<char>>& board, string word) {
	vector<string> ret;
	int len = word.length();
	char target = word[0];
	if (len > 1) {
		vector<string> childrenRet = myLoop2(board, word.substr(1, len - 1));
		int h = board.size(), w = board[0].size();
		vector<pair<int, int>> record{ {0,1},{1,0},{0,-1},{-1,0} };
		for (auto ad : childrenRet) {
			int lastI = ad[ad.length() - 2] - '0';
			int lastJ = ad[ad.length() - 1] - '0';
			for (auto addP : record) {
				int newI = lastI + addP.first;
				int newJ = lastJ + addP.second;
				if (newI >= 0 && newI < h && newJ >= 0 && newJ < w &&
					board[newI][newJ] == target) {
					string addStr = target + to_string(newI) + to_string(newJ);
					if (ad.find(addStr) == -1) {
						ret.push_back(ad + addStr);
					}
				}
			}
		}
	}
	else {
		for (int i = 0; i < board.size(); i++) {
			for (int j = 0; j < board[0].size(); j++) {
				// 如果字符匹配
				if (board[i][j] == target) {
					string addStr = target + to_string(i) + to_string(j);
					// 如果是最底层，则直接添加
					ret.push_back(addStr);
				}
			}
		}
	}
	return ret;
}
bool check(vector<vector<char>>& board, vector<vector<int>>& visited, int i, int j, string& s, int k) {
	if (board[i][j] != s[k]) {
		return false;
	}
	else if (k == s.length() - 1) {
		return true;
	}
	visited[i][j] = true;
	vector<pair<int, int>> directions{ {0, 1}, {0, -1}, {1, 0}, {-1, 0} };
	bool result = false;
	for (const auto& dir : directions) {
		int newi = i + dir.first, newj = j + dir.second;
		if (newi >= 0 && newi < board.size() && newj >= 0 && newj < board[0].size()) {
			if (!visited[newi][newj]) {
				bool flag = check(board, visited, newi, newj, s, k + 1);
				if (flag) {
					result = true;
					break;
				}
			}
		}
	}
	visited[i][j] = false;
	return result;
}
/*
给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。
单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

示例 1：
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true

示例 2：
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE"
输出：true

示例 3：
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCB"
输出：false

提示：
m == board.length
n = board[i].length
1 <= m, n <= 6
1 <= word.length <= 15
board 和 word 仅由大小写英文字母组成

进阶：你可以使用搜索剪枝的技术来优化解决方案，使其在 board 更大的情况下可以更快解决问题？

备注：
暴力破解法：当全为A，且很多时候，查找BnA(n个A)的时候，就会非常耗时。不过该方法可返回所有可行路径及其路径角标记录。
官方示例方法：通过遍历二维数组，当以此为起点时，是否能递归在上下左右长度1的距离找到下一个值，可以则继续，直到找完为止或者找到为止。通过二维数组记录前面哪些位置已经被访问。
搜索剪枝的技术优化方案：可以通过计算传入单词中的字符在集合中出现频次，先从出现频次较少的位置开始遍历，减少递归深度。
*/
bool Top100Liked::Solution::exist(vector<vector<char>>& board, string word, bool isMyOwn)
{
	unordered_map<char, int> cnt;
	// 统计集合中字符出现次数
	for (auto& row : board) {
		for (char c : row) {
			cnt[c]++;
		}
	}

	// 如果字符出现的次数比集合中还多则不可能存在路径
	unordered_map<char, int> word_cnt;
	for (char c : word) {
		if (++word_cnt[c] > cnt[c]) {
			return false;
		}
	}

	// 如果集合中出现头部的频次比尾部还多，则翻转字符串
	if (cnt[word.back()] < cnt[word[0]]) {
		reverse(word.begin(), word.end());
	}

	if (isMyOwn) {
		vector<string> address = myLoop2(board, word);
		return address.size() > 0;
	}
	else {
		int h = board.size(), w = board[0].size();
		vector<vector<int>> visited(h, vector<int>(w));
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				bool flag = check(board, visited, i, j, word, 0);
				if (flag) {
					return true;
				}
			}
		}
		return false;
	}
}

string myLoop3(string s, int times) {
	string tempRes;
	string num;
	string rep;
	int diff = 0;
	for (int i = 0; i < s.length(); i++) {
		if (s[i] == '[') {
			if (diff > 0) {
				rep.push_back(s[i]);
			}
			diff++;
		}
		else if (s[i] == ']') {
			diff--;
			if (diff > 0) {
				rep.push_back(s[i]);
			}
			if (diff == 0) {
				tempRes += myLoop3(rep, stoi(num));
				num.clear();
				rep.clear();
			}
		}
		else if (diff > 0) {
			rep.push_back(s[i]);
		}
		else if (diff == 0) {
			if (s[i] >= '0' && s[i] <= '9') {
				num.push_back(s[i]);
			}
			else {
				tempRes.push_back(s[i]);
			}
		}
	}
	string result;
	for (int i = 0; i < times; ++i) {
		result += tempRes;
	}
	return result;
}
/*
给定一个经过编码的字符串，返回它解码后的字符串。
编码规则为: k[encoded_string]，表示其中方括号内部的 encoded_string 正好重复 k 次。注意 k 保证为正整数。
你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。
此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 k ，例如不会出现像 3a 或 2[4] 的输入。

示例 1：
输入：s = "3[a]2[bc]"
输出："aaabcbc"

示例 2：
输入：s = "3[a2[c]]"
输出："accaccacc"

示例 3：
输入：s = "2[abc]3[cd]ef"
输出："abcabccdcdcdef"

示例 4：
输入：s = "abc3[cd]xyz"
输出："abccdcdcdxyz"

提示：
1 <= s.length <= 30
s 由小写英文字母、数字和方括号 '[]' 组成
s 保证是一个 有效 的输入。
s 中所有整数的取值范围为 [1, 300]

备注：
官方方法也是类似实现
*/
string Top100Liked::Solution::decodeString(string s, bool isMyOwn)
{
	isMyOwn = true;
	if (isMyOwn) {
		return myLoop3(s, 1);
	}
	else {
		return string();
	}
}





