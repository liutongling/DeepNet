import numpy as np
import matplotlib.pyplot as plt
import os
from openai import OpenAI
from experimental import *
# deepseek.py

import requests
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
# def h(x):
#     return x>0
#
#
# # 填写你的 API Key
# API_KEY = "sk-0db9cc1980db4bae9dd693791517a82a"
#
# url = "https://api.deepseek.com/chat/completions"
# headers = {
#     "Content-Type": "application/json",
#     "Authorization": f"Bearer {API_KEY}"
# }
#
# data = {
#     "model": "deepseek-reasoner",  # 指定使用 R1 模型（deepseek-reasoner）或者 V3 模型（deepseek-chat）
#     "messages": [
#         {"role": "system", "content": "你是一个专业的助手"},
#         {"role": "user", "content": "你是谁？"}
#     ],
#     "stream": False  # 关闭流式传输
# }
#
# response = requests.post(url, headers=headers, json=data)
def f(x):
    return 0.5*x*x+x+3

def numerical_diff(x):
    h = 10e-10
    print(h)
    print(f(x+h))
    return (f(x+h)-f(x))/h


def Test():
    s = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    t = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    batch = np.random.choice(20,5)
    print(batch)

    print(s[batch])
    #print(s[np.arange(20),t])


# if __name__ == '__main__':
#     # if response.status_code == 200:
#     #     result = response.json()
#     #     print(result['choices'][0]['message']['content'])
#     # else:
#     #     print("请求失败，错误码：", response.status_code)
#     #Test()
#     numerical_gradient(function,np.array([3.0,4.0]))
#     s = Sigmod()
#     s.a = 1
#     s.backward(1)
#     nt = np.random.rand(2,3)
#     print(nt)
#     print(type(nt))
#     s = np.array([[2,2,2],[2,2,2]])
#     t = np.array([[2,2,2],[2,2,2]])
#     print(s*t)
