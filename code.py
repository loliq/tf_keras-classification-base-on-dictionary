#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/31 22:51
# @Author  : LLL
# @Site    : 
# @File    : code.py
# @Software: PyCharm

def threeSumClosest(nums, target):
    if len(nums) < 3:
        return
        # 第二种，双指针优化
    nums.sort()
    res = nums[0] + nums[1] + nums[2]
    for i in range(len(nums) - 2):
        ## 求n数之和通用 重复的元素可以跳过
        if i > 1 and nums[i] == nums[i - 1]:
            continue
        left = i + 1
        right = len(nums) - 1
        while left < right:
            sum_i = nums[i] + nums[left] + nums[right]
            if sum_i == target:
                return target
            if abs(sum_i - target) < abs(res - target):
                res = sum_i
            if sum_i < target:
                left += 1
                # 元素剪枝
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
            elif sum_i > target:
                right -= 1
                # 元素相同剪枝
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
    return res
if __name__ == '__main__':
    nums = [-1, 0, 1, 1, 55]
    t = 3
    threeSumClosest(nums,3)