##当一个列表包含很多数据的时候，程序会一次加载所有数据到内存里边，所以程序会很慢，可以使用生成器来解决这个问题
##生成器不会一次性存储所有值，只有在使用到的时候，才生成一个值
def square_numbers(nums):
    result = []
    for i in nums:
        result.append(i * i)
    return  result

my_nums = square_numbers([1,2,3,4,5])
print(my_nums)
##定义生成器的两种方式
##1.生成器函数,使用yield关键字
def square_numbers1(nums):
    for i in nums:
        yield i * i

my_gen = square_numbers1([1,2,3,4,5])
##输出生成器的地址

print(my_gen)
##通过next函数输出生成器里边元素具体的值
print('Running next function')
print(next(my_gen))
##第一次调用next，只生成第一个值，也就是1

##通过循环输出生成器里边元素具体的值
print('Running for loop')
for num in my_gen:
    print(num)


##2.生成器表达式
my_list = [i*i for i in [1,2,3,4,5]]
print(my_list)
my_list = (i*i for i in [1,2,3,4,5])

