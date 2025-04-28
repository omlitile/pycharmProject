class Student:
    student_num = 0

    ##构造方法
    def __init__(self,name,sex):
        self.name = name
        self.sex = sex
        Student.student_num += 1
    ##类方法
    @classmethod
    def add_student(cls,add_num):
        cls.student_num += add_num

    @classmethod
    def form_string(cls,info):
        name,sex = info.split(' ')
        return cls(name,sex)

    ##静态方法
    @staticmethod
    def name_lenth(name):
        return len(name)


s1  = Student('kk','F')
s2 =  Student.form_string('kk F')
print(f's2.name: {s2.name},s2.name_len: {Student.name_lenth(s2.name)}')
print(f'Student.student_num： {Student.student_num}')
print(f's1.student_num: {s1.student_num}')
