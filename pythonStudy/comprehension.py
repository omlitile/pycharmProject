nums = [0,1,2,3,4,5]

my_list = []
for i in nums:
    my_list.append(i)
print(my_list)


my_list = [i for i in nums]
print(my_list)

################################################
##每个数变成各自的平方
nums = [0,1,2,3,4,5]

my_list = []
for i in nums:
    my_list.append(i**2)
print(my_list)


my_list = [i**2 for i in nums]
print(my_list)

###############################################
##如果为偶尔，则变成自己的平方
nums = [0,1,2,3,4,5]

my_list = []
for i in nums:
    if i % 2 == 0:
        my_list.append(i**2)
print(my_list)


my_list = [i**2 for i in nums if i%2==0]
print(my_list)

#########################################
## 组合两个列表
nums = [0,1,2,3,4,5]
letter = ['a','b','c']

my_list = []
for i in letter:
    for j in nums:
        my_list.append((i,j))
print(my_list)


my_dict = {}
for i,j in zip(letter,nums):
    my_dict[i] = j
print(my_dict)

my_dict = {i:j for i, j in zip(letter,nums)}
print(my_dict)

##########################################
##集合

l = [1,2,3,4,5,6,7,8,9]
my_set = set()
for i in l:
    my_set.add(i)
print(l)

my_set = {i for i in l}
print(my_set)
