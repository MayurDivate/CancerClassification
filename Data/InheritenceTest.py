class Person:
    def __init__(self, name):
        self.name = name

    def what_is_your_name(self):
        print(f'My name is {self.name}')

    def test(self):
        print(self.age)
        print(self.schoolname)


class Student(Person):
    def __init__(self, name, schoolname):
        self.name = name
        self.schoolname = schoolname

    def what_is_your_school_name(self):
        print(f'My school is {self.schoolname}')


class Teacher(Person):
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def what_is_your_age(self):
        print(f'I am {self.age} years old!')


P1 = Person('Mayur')
S1 = Student('ABC', 'XYZ')
T1 = Teacher('PQRS', 20)

P1.what_is_your_name()

S1.what_is_your_name()
S1.what_is_your_school_name()

T1.what_is_your_name()
T1.what_is_your_age()
T1.test()
