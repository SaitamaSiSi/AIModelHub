#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <set>
#include <unordered_set>
#include <algorithm>

using namespace std;

namespace Top100Liked
{
	class Solution
	{
	public:

#pragma region 哈希

		/// <summary>
		/// 两数之和
		/// </summary>
		/// <param name="nums"></param>
		/// <param name="target"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		vector<int> twoSum(vector<int>& nums, int target, bool isMyOwn = false);

		/// <summary>
		/// 字母异位词分组
		/// </summary>
		/// <param name="strs"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		vector<vector<string>> groupAnagrams(vector<string>& strs, bool isMyOwn = false);

		/// <summary>
		/// 最长连续序列
		/// </summary>
		/// <param name="nums"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		int longestConsecutive(vector<int>& nums, bool isMyOwn = false);

#pragma endregion

#pragma region 双指针

		/// <summary>
		/// 移动零
		/// </summary>
		/// <param name="nums"></param>
		/// <param name="isMyOwn"></param>
		void moveZeroes(vector<int>& nums, bool isMyOwn = false);

		/// <summary>
		/// 盛最多水的容器
		/// </summary>
		/// <param name="height"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		int maxArea(vector<int>& height, bool isMyOwn = false);

		/// <summary>
		/// 三数之和（未完成）
		/// </summary>
		/// <param name="nums"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		vector<vector<int>> threeSum(vector<int>& nums, bool isMyOwn = false);

		/// <summary>
		/// 接雨水
		/// </summary>
		/// <param name="height"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		int trap(vector<int>& height, bool isMyOwn = false);


#pragma endregion

#pragma region 滑动窗口

		/// <summary>
		/// 无重复字符的最长子串
		/// </summary>
		/// <param name="s"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		int lengthOfLongestSubstring(string s, bool isMyOwn = false);

		/// <summary>
		/// 找到字符串中所有字母异位词（超出时间限制）
		/// </summary>
		/// <param name="s"></param>
		/// <param name="p"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		vector<int> findAnagrams(string s, string p, bool isMyOwn = false);

#pragma endregion

#pragma region 子串

		/// <summary>
		/// 和为 K 的子数组（未实现）
		/// </summary>
		/// <param name="nums"></param>
		/// <param name="k"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		int subarraySum(vector<int>& nums, int k, bool isMyOwn = false);

		/// <summary>
		/// 滑动窗口最大值
		/// </summary>
		/// <param name="nums"></param>
		/// <param name="k"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		vector<int> maxSlidingWindow(vector<int>& nums, int k, bool isMyOwn = false);

		/// <summary>
		/// 最小覆盖子串（参考）
		/// </summary>
		/// <param name="s"></param>
		/// <param name="t"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		string minWindow(string s, string t, bool isMyOwn = false);

#pragma endregion

#pragma region 普通数组



#pragma endregion

	};
}


