import math
import random
from functools import reduce
import machine

def ndim(arry):
    if type(arry) == int or type(arry) == float:
        return 0
    return 1 + ndim(arry[0])

def element_length(arry):
    count = 0
    if ndim(arry) == 0:
        return 1
    if ndim(arry) == 1:
        return len(arry)
    for n in arry:
        count += element_length(n)
    return count

def flatten(arry, only_1dim=False):
    if ndim(arry) <= 1:
        return arry
    result = []
    for n in arry:
        if ndim(n) == 1:
            result += n
        else:
            if only_1dim:
                result.append(flatten(n))
            else:
                result += flatten(n)
    return result

def T(arry):
    if ndim(arry) == 0:
        return arry
    transposed = []
    if ndim(arry) == 1:
        for n in arry:
            transposed.append(n)
        return transposed
    for i in range(len(arry[0])):
        row = []
        for j in range(len(arry)):
            row.append(arry[j][i])
        transposed.append(row)
    return transposed

def reshape(arry, row, col):
    out_row = []
    for i in range(row):
        out_col = []
        for j in range(col):
            out_col.append(arry[i * col + j])
        out_row.append(out_col)
    return out_row

def copy(arry):
    if ndim(arry) == 0:
        return arry
    if ndim(arry) == 1:
        return arry.copy()
    copied = []
    for n in arry:
        copied.append(copy(n))
    return copied

def zeros(arry):
    if type(arry) == int:
        return [0 for n in range(arry)]
    else:
        return [[0 for n in range(arry[1])] for n in range(arry[0])]

def random_seed(num=False):
    if num != False:
        random.seed(num)
        return
    TEMPERATURE_SENSOR = machine.ADC(4)
    seed = TEMPERATURE_SENSOR.read_u16() % 100
    print("random_seed=", seed)
    random.seed(seed)
    return

def random_list(row, col):
    return [[random.random() for n in range(col)] for n in range(row)]

def count_nonzero(arry):
    lists = flatten(arry)
    count = 0
    for n in lists:
        if n != 0:
            count += 1
    return count

def count_num(arry, num):
    lists = flatten(arry)
    count = 0
    for n in lists:
        if n == num:
            count += 1
    return count

def diag(arry):
    return [arry[n][n] for n in [0, 1, 2]]

def diag_fliplr(arry):
    return [arry[n][2-n] for n in [0, 1, 2]]

def add(arry1, arry2):
    if ndim(arry1) == 0 and ndim(arry2) == 0:
        return arry1 + arry2
    if ndim(arry2) == 0:
        return [add(n,arry2) for n in arry1]
    if ndim(arry1) == 0:
        return [add(arry1,n) for n in arry2]
    result = []
    for n1, n2 in zip(arry1, arry2):
        result.append(add(n1, n2))
    return result

def sub(arry1, arry2):
    if ndim(arry1) == 0 and ndim(arry2) == 0:
        return arry1 - arry2
    if ndim(arry2) == 0:
        return [sub(n, arry2) for n in arry1]
    if ndim(arry1) == 0:
        return [sub(arry1, n) for n in arry2]
    result = []
    for n1, n2 in zip(arry1, arry2):
        result.append(sub(n1, n2))
    return result

def multiple(arry1, arry2):
    if ndim(arry1) == 0 and ndim(arry2) == 0:
        return arry1 * arry2
    if ndim(arry2) == 0:
        return [multiple(n,arry2) for n in arry1]
    if ndim(arry1) == 0:
        return [multiple(arry1,n) for n in arry2]
    if ndim(arry1) == 1 and ndim(arry2) == 1:
        return [n1 * n2 for n1, n2 in zip(arry1, arry2)]
    if ndim(arry1) == 1:
        return [multiple(arry1, n) for n in arry2]
    if ndim(arry2) == 1:
        return [multiple(n, arry2) for n in arry1]
    return [multiple(n1, n2) for n1,n2 in zip(arry1,arry2)]

def divide(arry, num):
    if type(arry) == int or type(arry) == float:
        return arry / num
    return [divide(n,num) for n in arry]

def exp(arry):
    if type(arry) == int or type(arry) == float:
        return math.exp(arry)
    return [exp(n) for n in arry]

def reversal_of_numerator(arry):
    if type(arry) == int or type(arry) == float:
        return 1.0 / arry
    return [reversal_of_numerator(n) for n in arry]

def my_min(arry):
    min_num = 9999999
    for n in arry:
        if ndim(n) == 0:
            if n < min_num:
                min_num = n
        else:
            res = my_min(n)
            if min_num > res:
                min_num = res
    return min_num

def my_max(arry):
    max_num = -9999999
    for n in arry:
        if ndim(n) == 0:
            if n > max_num:
                max_num = n
        else:
            res = my_max(n)
            if max_num < res:
                max_num = res
    return max_num

def dot(arry1, arry2):
    if ndim(arry1) == 0 or ndim(arry2) == 0:
        return arry1 * arry2
    if ndim(arry1) == 1 and ndim(arry2) == 0:
        multiple_arry = multiple(arry1, arry2)
        return reduce(lambda x, y: x + y, multiple_arry)
    if ndim(arry1) == 0 and ndim(arry2) == 1:
        multiple_arry = multiple(arry2, arry1)
        return reduce(lambda x, y: x + y, multiple_arry)
    if ndim(arry1) == 1 and ndim(arry2) == 1:
        ans = 0
        if len(arry1) != len(arry2):
            raise ValueError("array1 and array2 must have same length")
        for n in range(len(arry1)):
            ans += arry1[n] * arry2[n]
        return ans
    if ndim(arry1) == 1:
        return [dot(arry1, n) for n in T(arry2)]
    if ndim(arry2) == 1:
        return [dot(n, arry2) for n in arry1]
    return [[dot(n1, n2) for n2 in T(arry2)] for n1 in arry1]

def outer(arry1, arry2):
    list1 = flatten(arry1)
    list2 = flatten(arry2)
    if ndim(list1) == 0 and ndim(list1) == 0:
        return list1 * list2
    if ndim(list1) == 0 or ndim(list1) == 0:
        return multiple(list1, list2)
    result = [[i * j for j in list2] for i in list1]
    return result

def mean(arry, axis=None):
    if axis == None:
        lists = flatten(arry)
        return sum(lists) / len(lists)
    if axis == 0:
        return [sum(n) / len(n) for n in T(arry)]
    if axis == 1:
        return [sum(n) / len(n) for n in arry]
    return None

def argmax(arry):
    lists = flatten(arry)
    max = lists[0]
    max_index = 0
    for i in range(len(lists)):
        if lists[i] > max:
            max = lists[i]
            max_index = i
    return max_index

def clip(arry, a_min, a_max):
    if ndim(arry) == 0:
        if arry < a_min:
            return a_min
        if arry > a_max:
            return a_max
        return arry
    return [clip(n, a_min, a_max) for n in arry]
