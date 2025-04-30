

def add(a,b):
    return a + b
print(add(3,4))

##使用lambda关键字来创建匿名函数
#方法名字 add
#lambda  关键字
# a,b   参数，如果无参可以省略
#:a+b   返回值
add =  lambda a,b: a+b
print(add(3,4))


my_list = [1,2,3,4,5]
new_list = list(map(lambda x: x**2 ,my_list))
print(new_list)

##使用lambda函数优化if else
##根据用户等级给予用户积分
def user_login(user):
    if user.level == 1:
        user.credits += 2
    elif user.level == 2:
        user.credits += 5
    elif user.level == 3:
        user.credits +=10
    elif user.level == 4:
        user.credits += 20

def user_login1(user):
    level_credit_map = {
        1: lambda  x:x+2,
        2: lambda  x:x+5,
        3: lambda  x:x+10,
        4: lambda  x:x+20
    }
    user.credits = level_credit_map[user.level](user.credits)
