#bubble sort
def bubble_sort(nums):
    for i in range(len(nums)):
        for j in range(1,len(nums)):
            if nums[j-1] > nums[j]:
                nums[j],nums[j-1] = nums[j-1],nums[j]
    return nums

if __name__ =="__main__":
    a = [6,7,9,14,2,8,1]
    print(bubble_sort(a))