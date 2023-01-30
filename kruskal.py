## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##      http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
##
## Author: Richard Sproat (rws@xoba.com)
"""Edit distance.
"""

import sys

## `Kruskal` algorithm

class Cell:
    def __init__(self, cost, bakptr, elt1, elt2):
        self.cost_ = cost
        self.bakptr_ = bakptr
        self.elt1_ = elt1
        self.elt2_ = elt2
        return
    def Cost(self): return self.cost_
    def BackPointer(self): return self.bakptr_
    def Pair(self): return (self.elt1_, self.elt2_)

def DefaultIns(item):
    return 1
def DefaultDel(item):
    return 1
def DefaultSub(item1, item2):
    if item1 == item2: return 0
    return 1

class Matrix:
    def __init__(self, list1, list2):
        list1 = [None] + list(list1)
        list2 = [None] + list(list2)
        self.max1_ = len(list1)
        self.max2_ = len(list2)
        self.data_ = {}
        self.data_[(0,0)] = Cell(0,None,None,None)
        cum = 0
        for i in range(1, self.max1_):
            cum += INS_(list1[i])
            self.data_[(i,0)] = Cell(cum,self.data_[(i-1,0)],list1[i],None)
        cum = 0
        for i in range(1, self.max2_):
            cum += DEL_(list2[i])
            self.data_[(0,i)] = Cell(cum,self.data_[(0,i-1)],None,list2[i])
        for i in range(1, self.max1_):
            for j in range(1, self.max2_):
                l1el = list1[i]
                l2el = list2[j]
                c1 = self.data_[(i, j-1)].Cost() + INS_(l1el)
                c2 = self.data_[(i-1, j)].Cost() + DEL_(l2el)
                c3 = self.data_[(i-1, j-1)].Cost() + SUB_(l1el,l2el)
                if c1 <= c2 and c1 <= c3:
                    self.data_[(i,j)] = Cell(c1, self.data_[(i, j-1)], None, l2el)
                elif c2 <= c1 and c2 <= c3:
                    self.data_[(i,j)] = Cell(c2, self.data_[(i-1, j)], l1el, None)
                else:
                    self.data_[(i,j)] = Cell(c3, self.data_[(i-1, j-1)], l1el, l2el)
        c = self.data_[(self.max1_-1, self.max2_-1)]
        path = []
        while c:
            if not c.Pair() == (None, None):
                path.append(c.Pair())
            c = c.BackPointer()
        path.reverse()
        self._path = path
        return
    def path(self):
        return self._path

def BestMatch (l1, l2):
    m = Matrix(l1, l2)
    cost = m.data_[(m.max1_-1, m.max2_-1)].cost_
    return cost, m.path()


# ReadTable()

# def Insert(a):
#     try: return(TABLE_[(None, a)])
#     except KeyError: return DefaultIns(a)

# def Delete(a):
#     try: return(TABLE_[(a, None)])
#     except KeyError: return DefaultDel(a)

# def Substitute(a, b):
#     try: return(TABLE_[(a, b)])
#     except KeyError: return DefaultSub(a, b)

INS_ = DefaultIns
DEL_ = DefaultDel
SUB_ = DefaultSub


### Hack to speed things up a bit:

def LCPRemainders(s1, s2):
    len1 = len(s1)
    len2 = len(s2)
    offset = 0
    while (offset < len1 and offset < len2):
        if s1[offset] != s2[offset]: break
        offset += 1
    return s1[offset:], s2[offset:]


def LCSRemainders(s1, s2):
    len1 = len(s1)-1
    len2 = len(s2)-1
    while (len1 > -1 and len2 > -1):
        if s1[len1] != s2[len2]: break
        len1 -= 1
        len2 -= 1
    return s1[:len1+1], s2[:len2+1]
