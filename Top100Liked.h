#pragma once

#include <set>
#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <algorithm> // ���� std::sort
// #include <numeric> // ���� std::gcd

using namespace std;

namespace Top100Liked
{
	class Solution
	{
	public:

#pragma region ��ϣ

		/// <summary>
		/// ����֮��
		/// </summary>
		/// <param name="nums"></param>
		/// <param name="target"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		vector<int> twoSum(vector<int>& nums, int target, bool isMyOwn = false);

		/// <summary>
		/// ��ĸ��λ�ʷ���
		/// </summary>
		/// <param name="strs"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		vector<vector<string>> groupAnagrams(vector<string>& strs, bool isMyOwn = false);

		/// <summary>
		/// ���������
		/// </summary>
		/// <param name="nums"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		int longestConsecutive(vector<int>& nums, bool isMyOwn = false);

#pragma endregion

#pragma region ˫ָ��

		/// <summary>
		/// �ƶ���
		/// </summary>
		/// <param name="nums"></param>
		/// <param name="isMyOwn"></param>
		void moveZeroes(vector<int>& nums, bool isMyOwn = false);

		/// <summary>
		/// ʢ���ˮ������
		/// </summary>
		/// <param name="height"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		int maxArea(vector<int>& height, bool isMyOwn = false);

		/// <summary>
		/// ����֮�ͣ�δ��ɣ�
		/// </summary>
		/// <param name="nums"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		vector<vector<int>> threeSum(vector<int>& nums, bool isMyOwn = false);

		/// <summary>
		/// ����ˮ
		/// </summary>
		/// <param name="height"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		int trap(vector<int>& height, bool isMyOwn = false);


#pragma endregion

#pragma region ��������

		/// <summary>
		/// ���ظ��ַ�����Ӵ�
		/// </summary>
		/// <param name="s"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		int lengthOfLongestSubstring(string s, bool isMyOwn = false);

		/// <summary>
		/// �ҵ��ַ�����������ĸ��λ�ʣ�����ʱ�����ƣ�
		/// </summary>
		/// <param name="s"></param>
		/// <param name="p"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		vector<int> findAnagrams(string s, string p, bool isMyOwn = false);

#pragma endregion

#pragma region �Ӵ�

		/// <summary>
		/// ��Ϊ K �������飨δʵ�֣�
		/// </summary>
		/// <param name="nums"></param>
		/// <param name="k"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		int subarraySum(vector<int>& nums, int k, bool isMyOwn = false);

		/// <summary>
		/// �����������ֵ
		/// </summary>
		/// <param name="nums"></param>
		/// <param name="k"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		vector<int> maxSlidingWindow(vector<int>& nums, int k, bool isMyOwn = false);

		/// <summary>
		/// ��С�����Ӵ����ο���
		/// </summary>
		/// <param name="s"></param>
		/// <param name="t"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		string minWindow(string s, string t, bool isMyOwn = false);

#pragma endregion

#pragma region ��ͨ����

		/// <summary>
		/// ����������
		/// </summary>
		/// <param name="nums"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		int maxSubArray(vector<int>& nums, bool isMyOwn = false);

		/// <summary>
		/// �ϲ����䣨�ο���
		/// </summary>
		/// <param name="intervals"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		vector<vector<int>> merge(vector<vector<int>>& intervals, bool isMyOwn = false);

		/// <summary>
		/// ��ת����
		/// </summary>
		/// <param name="nums"></param>
		/// <param name="k"></param>
		/// <param name="isMyOwn"></param>
		void rotate(vector<int>& nums, int k, bool isMyOwn = false);

		/// <summary>
		/// ��������������ĳ˻�
		/// </summary>
		/// <param name="nums"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		vector<int> productExceptSelf(vector<int>& nums, bool isMyOwn = false);

		/// <summary>
		/// ȱʧ�ĵ�һ������
		/// </summary>
		/// <param name="nums"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		int firstMissingPositive(vector<int>& nums, bool isMyOwn = false);

#pragma endregion

#pragma region ����

		/// <summary>
		/// ��������
		/// </summary>
		/// <param name="matrix"></param>
		/// <param name="isMyOwn"></param>
		void setZeroes(vector<vector<int>>& matrix, bool isMyOwn = false);

		/// <summary>
		/// ��������
		/// </summary>
		/// <param name="matrix"></param>
		/// <param name="isMyOwn"></param>
		/// <returns></returns>
		vector<int> spiralOrder(vector<vector<int>>& matrix, bool isMyOwn = false);

#pragma endregion

	};
}


