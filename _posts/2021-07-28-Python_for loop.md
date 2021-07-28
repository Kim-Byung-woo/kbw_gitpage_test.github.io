---
title:  "[Python] For loop"
excerpt: "Pthon function tutorial"
toc: true
toc_sticky: true
header:
  teaser: /assets/images/teaser.png

categories:
  - pythonic

tags:
  - python
  - basic
  - function
  - loop

author_profile: true
sidebar:
  nav: "sidebar-contents"
---

# for Loop

for 변수 in 리스트(또는 튜플, 문자열):
<br/>
　　수행할 문장1
<br/>
　　수행할 문장2   


```python
test_list = ['one', 'two', 'three']
for idx in test_list:
    print(f'{idx}')
```

    one
    two
    three
    

# continue
for문 안의 문장을 수행하는 도중에 continue문을 만나면 for문의 처음으로 돌아가게 된다.


```python
marks = [90, 25, 67, 45, 80]
number = 0 
for mark in marks: 
    number = number +1 
    if mark < 60:
        continue 
    print("%d번 학생 축하합니다. 합격입니다. " % number)
```

# range


```python
marks = [90, 25, 67, 45, 80]
number = 0 
for number in range(len(marks)): # range(len(marks)) = range(5)
    if marks[number] < 60:
        continue 
    print("%d번 학생 축하합니다. 합격입니다. " % (number + 1))  # %뒤에 오는  number는 문자형으로 인식되기 때문에 괄호로 묵어서 사용합니다.

```

    1번 학생 축하합니다. 합격입니다. 
    3번 학생 축하합니다. 합격입니다. 
    5번 학생 축하합니다. 합격입니다. 
    

# enumerate
enumerate는 순서가 있는 자료형(리스트, 튜플, 문자열)을 입력으로 받아 인덱스 값을 포함하는 enumerate 객체를 돌려준다.


```python
for number, mark in enumerate(marks): # number: index를 저장할 변수, mark 요소를 저장할 변수
    if mark < 60:
        continue 
    print("%d번 학생 축하합니다. 합격입니다. " % (number + 1))  # %뒤에 오는  number는 문자형으로 인식되기 때문에 괄호로 묵어서 사용합니다.
```

    1번 학생 축하합니다. 합격입니다. 
    3번 학생 축하합니다. 합격입니다. 
    5번 학생 축하합니다. 합격입니다. 
    


```python
num = int(input('정수를 입력해주세요: '))
prime_num = True
for idx in range(2, num):
    if num % idx == 0:
        prime_num = False
        break

if  prime_num == True:
    print(f'{num}는 소수입니다.')
else:
    print(f'{num}는 소수가 아닙니다.')
```

    정수를 입력해주세요:  21
    

    21는 소수가 아닙니다.
    

# List Comprehension
+ format: [출력표현식 for 요소 in 입력 Sequence]
이미 생성된 리스트를 사용해서 새로운 리스트를 생성할때 사용합니다.


```python
a = [1, 3, 5, 7, 9]
result = [num * 3 for num in a] # a의 요소들을 num*3으로 출력
print(result)
```

    [3, 9, 15, 21, 27]
    

## List Comprehension if
+ format: [출력표현식 if 조건식 else 출력표현식 for 요소 in 입력 Sequence]


```python
a = [1, 3, 5, 7, 9]
result = [i * 4 if i > 2 else i * 5 for i in a] # 조건절 충족시 출력표현식 사용 조건 미충족시 else절 사용
print(result)
```

    [5, 12, 20, 28, 36]
    

## List Comprehension if non else
+ format: [출력표현식 if 조건식 else 출력표현식 for 요소 in 입력 Sequence] else가 없을 때는 if를 뒤에 써야 한다.


```python
a = [1, 2, 5, 6, 9, 10]
result = [num * 3 for num in a if num % 2 == 0]
print(result)
```

    [6, 18, 30]
    

# dictionary Comprehension 
+ format: 출력 표현식에 ':'를  추가합니다.


```python
a = [1, 2, 3, 4]
d = {i : i * 3 for i in a}
print(d)
```

    {1: 3, 2: 6, 3: 9, 4: 12}
    
