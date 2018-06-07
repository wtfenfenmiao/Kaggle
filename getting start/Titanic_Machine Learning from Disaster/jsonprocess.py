#coding:utf-8
import json
f=open('wt.demo.test.json','r',encoding='UTF-8')
question=[]
document=[]
answer=[]
for line in f.readlines():
    data=json.loads(line)
    for para in data['documents']:
        del para['segmented_title']
        del para['segmented_paragraphs']
        document.append(para)
    question.append(data['question'])

ans=open('wt.demo.predict.json','r',encoding='UTF-8')
for line in ans.readlines():
    data=json.loads(line)
    answer.append(data['answers'])

print ("*************************************")
print ("问题和答案\n")
for (qu,ans) in zip(question,answer):
    print ("问题："+qu)
    print ("答案："+ans[0]+"\n")

print ("*************************************")
print ("候选文档集：\n")
for doc in document:
    print (doc['title'])
    print (doc['paragraphs'])
    print ("\n")






