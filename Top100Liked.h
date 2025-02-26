#pragma once

#include <set>
#include <map>
#include <stack>
#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <algorithm> // 包含 std::sort
// #include <numeric> // 包含 std::gcd

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

		/// <summary>
		/// 最大子数组和
		/// </summary>
		/// <param name="nums"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		int maxSubArray(vector<int>& nums, bool isMyOwn = false);

		/// <summary>
		/// 合并区间（参考）
		/// </summary>
		/// <param name="intervals"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		vector<vector<int>> merge(vector<vector<int>>& intervals, bool isMyOwn = false);

		/// <summary>
		/// 轮转数组
		/// </summary>
		/// <param name="nums"></param>
		/// <param name="k"></param>
		/// <param name="isMyOwn"></param>
		void rotate(vector<int>& nums, int k, bool isMyOwn = false);

		/// <summary>
		/// 除自身以外数组的乘积
		/// </summary>
		/// <param name="nums"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		vector<int> productExceptSelf(vector<int>& nums, bool isMyOwn = false);

		/// <summary>
		/// 缺失的第一个正数
		/// </summary>
		/// <param name="nums"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		int firstMissingPositive(vector<int>& nums, bool isMyOwn = false);

#pragma endregion

#pragma region 矩阵

		/// <summary>
		/// 矩阵置零
		/// </summary>
		/// <param name="matrix"></param>
		/// <param name="isMyOwn"></param>
		void setZeroes(vector<vector<int>>& matrix, bool isMyOwn = false);

		/// <summary>
		/// 螺旋矩阵
		/// </summary>
		/// <param name="matrix"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		vector<int> spiralOrder(vector<vector<int>>& matrix, bool isMyOwn = false);

		/// <summary>
		/// 旋转图像
		/// </summary>
		/// <param name="matrix"></param>
		/// <param name="isMyOwn"></param>
		void rotate(vector<vector<int>>& matrix, bool isMyOwn = false);

		/// <summary>
		/// 搜索二维矩阵 II
		/// </summary>
		/// <param name="matrix"></param>
		/// <param name="target"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		bool searchMatrix(vector<vector<int>>& matrix, int target, bool isMyOwn = false);

#pragma endregion

#pragma region 链表

		struct ListNode {
			int val;
			ListNode* next;
			ListNode(int x) : val(x), next(NULL) {}

		};
		/// <summary>
		/// 160 相交链表（未实现）
		/// </summary>
		/// <param name="headA"></param>
		/// <param name="headB"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		ListNode* getIntersectionNode(ListNode* headA, ListNode* headB, bool isMyOwn = false);

		struct ListNode2 {
			int val;
			ListNode2* next;
			ListNode2() : val(0), next(nullptr) {}
			ListNode2(int x) : val(x), next(nullptr) {}
			ListNode2(int x, ListNode2* next) : val(x), next(next) {}
		};
		/// <summary>
		/// 反转链表
		/// </summary>
		/// <param name="head"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		ListNode2* reverseList(ListNode2* head, bool isMyOwn = false);

#pragma endregion

#pragma region 回溯

		/// <summary>
		/// 全排列
		/// </summary>
		/// <param name="nums"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		vector<vector<int>> permute(vector<int>& nums, bool isMyOwn = false);

		/// <summary>
		/// 括号生成
		/// </summary>
		/// <param name="n"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		vector<string> generateParenthesis(int n, bool isMyOwn = false);

		/// <summary>
		/// 单词搜索
		/// </summary>
		/// <param name="board"></param>
		/// <param name="word"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		bool exist(vector<vector<char>>& board, string word, bool isMyOwn = false);

#pragma endregion

#pragma region 栈

		/// <summary>
		/// 字符串解码
		/// </summary>
		/// <param name="s"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		string decodeString(string s, bool isMyOwn = false);

		/// <summary>
		/// 柱状图中最大的矩形（超时）
		/// </summary>
		/// <param name="heights"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		int largestRectangleArea(vector<int>& heights, bool isMyOwn = false);

#pragma endregion


	};
}


