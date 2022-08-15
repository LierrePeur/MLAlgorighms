import pandas as pd
import math
import numpy as np


def ID3(examples, attrs):
     root = Node()
     max_gain = 0
     max_feat = ""
     for feature in attrs:
        #print ("\n",examples)
        gain = info_gain(examples, feature)
        if gain > max_gain:
        max_gain = gain
        max_feat = feature
     root.value = max_feat
     #print ("\nMax feature attr",max_feat)
     uniq = np.unique(examples[max_feat])
     #print ("\n",uniq)
     for u in uniq:
        #print ("\n",u)
        subdata = examples[examples[max_feat] == u]
        #print ("\n",subdata)
        if entropy(subdata) == 0.0:
            newNode = Node()
            newNode.isLeaf = True
            newNode.value = u
            newNode.pred = np.unique(subdata["answer"])
            root.children.append(newNode)
        else:
            dummyNode = Node()
            dummyNode.value = u
            new_attrs = attrs.copy()
            new_attrs.remove(max_feat)
            child = ID3(subdata, new_attrs)
            dummyNode.children.append(child)
            root.children.append(dummyNode)
     return root


def entropy(examples):
     pos = 0.0
     neg = 0.0
     for _, row in examples.iterrows():
        if row["answer"] == "yes":
            pos += 1
        else:
            neg += 1
     if pos == 0.0 or neg == 0.0:
        return 0.0
     else:
         p = pos / (pos + neg)
         n = neg / (pos + neg)
         return -(p * math.log(p, 2) + n * math.log(n, 2))


def info_gain(examples, attr):
     uniq = np.unique(examples[attr])
     #print ("\n",uniq)
     gain = entropy(examples)
     #print ("\n",gain)
     for u in uniq:
        subdata = examples[examples[attr] == u]
        #print ("\n",subdata)
        sub_e = entropy(subdata)
        gain -= (float(len(subdata)) / float(len(examples))) * sub_e
        #print ("\n",gain)
     return gain
