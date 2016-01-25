# -*- coding: utf-8 -*- 

import os
import re
import shutil
import math
import simplejson as sj
# import numpy as np
import random as rd

"""
└┕┖┗
"""
"""
functions:
└-lda_model_parameter
└-lda_document_initiation
└-lda_inference
 └-in a single inference process,do Gibbs samling for each word in every document
└-lda_sampling
 └-chose a new topic word by word
└-lda_update_estimated_parameter
 └-update phi and theta
"""




class LdaModel:

    __alpha = 2.5  # usual value is 50 / K
    __beta = 0.100  # usual value is 0.1
    __topicNum = 20
    __iteration = 150
    __saveStep = 20
    __beginSaveIters = 50


    # one-line functions
    # private functions
    # using in the module

    # return a integer between 0 to topicNum-1
    def __getRandom(self):
        return int(rd.uniform(0, self.__topicNum))


    def __ModelParameter(self):
        pass


    def __get2DMatrix(self,x, y, default=0.0):
        return [[default for j in range(0, y)] for i in range(0, x)]

    # depend on file structure
    def __getDocWord(self,m, v):
        return self.__fileData['doc'][m]['words'][v]

    def __getM(self,FileData):
        return len(FileData['doc'])

    def __getV(self,FileData):
        return len(FileData['words'])

    def __getWordList(self,FileData):
        return FileData['words']

    def __getWordsInFileM(self,FileData,m):
        return FileData['doc'][m]['words']

    """
    number of words in doc m assigned to topic initTopic add 1:
    └-nmk[m][initTopic]
    number of terms doc[m][n] assigned to topic initTopic add 1:
    └-nkt[initTopic][doc[m][n]]
    total number of words assigned to topic initTopic add 1:
    └-nktSum[initTopic]
    total number of words in document m is N:
     └-nmkSum[m] = N
    """
    # global values
    __z = None
    __nmk = None
    __nkt = None
    __nktSum = None
    __nmkSum = None
    __M = None
    __V = None
    __K = None
    __fileData = None

    __phi = None
    __theta = None

    # fileData = None
    def readFile(self,filedir):
        self.__fileData = sj.loads(open(filedir, 'r').read())


    def DocumentInitiation(self):
        FileData = self.__fileData
        self.__M = self.__getM(FileData)
        self.__V = self.__getV(FileData)
        self.__K = self.__topicNum
        # nmk: 每篇文章中，每个种类主题下词汇的个数
        self.__nmk = self.__get2DMatrix(self.__M, self.__K)

        self.__phi = [{} for i in range(0, self.__K)]
        self.__theta = self.__get2DMatrix(self.__M, self.__K)
        # nkt: 每个主题中,某个词汇的数量
        self.__nkt = [{} for i in range(0, self.__K)]
        # nktSum: 每个主题下词汇的总数
        self.__nktSum = [0 for i in range(0, self.__K)]
        # nmkSum: 每篇文章的词汇数
        self.__nmkSum = [0 for i in range(0, self.__M)]
        self.__z = [[] for i in range(0, self.__M)]
        for m in range(0, self.__M):
            words = self.__getWordsInFileM(FileData,m)
            self.__nmkSum[m] = len(words)
            self.__z[m] = [1 for i in range(0, self.__nmkSum[m])]
            for v in range(0, self.__nmkSum[m]):
                topic = self.__getRandom()
                # topic
                self.__z[m][v] = topic
                self.__nmk[m][topic] += 1
                self.__nkt[topic][self.__getDocWord(m, v)] = self.__nkt[topic].get(self.__getDocWord(m, v), 0.0) + 1.0
                self.__nktSum[topic] += 1
        pass
        """
        dir_ = os.getcwd()+'\\lda_result'
        if os.path.exists(dir_)==False:
            os.mkdir(dir_)
        open(os.getcwd()+'\\lda_result\\nmk','w').write(sj.dumps(self.__nmk,indent=1).encode('utf-8'))
        open(os.getcwd()+'\\lda_result\\nkt','w').write(sj.dumps(self.__nkt,indent=1).encode('utf-8'))
        open(os.getcwd()+'\\lda_result\\nmkSum','w').write(sj.dumps(self.__nmkSum,indent=1).encode('utf-8'))
        open(os.getcwd()+'\\lda_result\\nktSum','w').write(sj.dumps(self.__nktSum,indent=1).encode('utf-8'))
        """

    def lda_inference(self):
        # check i
        if self.__iteration < (self.__saveStep + self.__beginSaveIters):
            print('iterations wrong')
            exit()

        for i in range(0, self.__iteration):
            print '%d inference...' % i
            # if i reach meet the conditions,store result
            if (i >= self.__beginSaveIters) & ((i - self.__beginSaveIters) % self.__saveStep == 0):
                print 'save at iteration',i
                self.__updateEstimatedParameters()
                self.__saveIteratedModel(i)
                
            # Gibbs sampling for each word word in ducuments
            count = 0
            for m in range(0, self.__M):
                for v in range(0, self.__nmkSum[m]):
                    count += 1
                    newType = self.__GibbsSamplingZ(m, v)
                    self.__z[m][v] = newType
                    print '%d words finished\r' % count,
            print
            # self.__updateEstimatedParameters()
            # self.__saveIteratedModel(i)
        print 'save the last inference...'
        self.__updateEstimatedParameters()
        self.__saveIteratedModel('LAST')
        self.__saveLast()

    def __GibbsSamplingZ(self,m, n):
        pass
        # Sample from p(z_i|z_-i, w) using Gibbs upde rule
        oldTopic = self.__z[m][n]
        self.__nmk[m][oldTopic] -= 1
        self.__nkt[oldTopic][self.__getDocWord(m, n)] -= 1
        self.__nmkSum[m] -= 1
        self.__nktSum[oldTopic] -= 1
        # compute p(z_i|z_-i, w)
        p = [(self.__nkt[k].get(self.__getDocWord(m, n),0) + self.__beta) / (self.__nktSum[k] + self.__V * self.__beta) * \
            (self.__nmk[m][k] + self.__alpha) / (self.__nmkSum[m] + self.__K * self.__alpha) \
            for k in range(0, self.__K)]
        # sampling
        for k in range(1, self.__K):
            p[k] += p[k - 1]
        u = rd.uniform(0, 1)
        # 掷针法
        newTopic = 0
        for i in range(0, self.__K):
            if u < (p[i]/p[self.__K - 1]):
                #print u,p[i]/p[self.__K - 1],i
                newTopic = i
                break
        # update global information
        self.__nmk[m][newTopic] += 1
        self.__nkt[newTopic][self.__getDocWord(m, n)]  = self.__nkt[newTopic].get(self.__getDocWord(m, n),0.0) + 1.0
        self.__nmkSum[m] += 1
        self.__nktSum[newTopic] += 1
        return newTopic;

    def __updateEstimatedParameters(self):
        for k in range(0,self.__K):
            for t in self.__getWordList(self.__fileData):
                self.__phi[k][t] = (self.__nkt[k].get(t,0) + self.__beta) / (self.__nktSum[k] + self.__V * self.__beta)
        for m in range(0,self.__M):
            for k in range(0,self.__K):
                self.__theta[m][k] = (self.__nmk[m][k] + self.__alpha) / (self.__nmkSum[m] + self.__K * self.__alpha)

    def __saveIteratedModel(self,i,mode='full',representativeItem=False):
        # save phi
        open(os.getcwd()+'\\lda_result\\phi_at_iteration_'+str(i),'w').write(sj.dumps(self.__phi,indent=1))
        # save theta
        open(os.getcwd()+'\\lda_result\\theta_at_iteration_'+str(i),'w').write(sj.dumps(self.__theta,indent=1))
        # save doc(m,v):z
        open(os.getcwd()+'\\lda_result\\z_at_iteration_'+str(i),'w').write(sj.dumps(self.__z,indent=1))
        # save nmk,nkt,nmkSum,nktSum
        if(mode=='full'):
            open(os.getcwd()+'\\lda_result\\nmk_'+str(i),'w').write(sj.dumps(self.__nmk,indent=1).encode('utf-8'))
            open(os.getcwd()+'\\lda_result\\nkt_'+str(i),'w').write(sj.dumps(self.__nkt,indent=1).encode('utf-8'))
            open(os.getcwd()+'\\lda_result\\nmkSum_'+str(i),'w').write(sj.dumps(self.__nmkSum,indent=1).encode('utf-8'))
            open(os.getcwd()+'\\lda_result\\nktSum_'+str(i),'w').write(sj.dumps(self.__nktSum,indent=1).encode('utf-8'))
        # save
        if (representativeItem==True):
            pass
        # save phi top  words
        fp = open(os.getcwd()+'\\lda_result\\topicWords_at_iteration_'+str(i),'w')
        for k in range(0,self.__K):
            topK = self.__phi[k]
            top10 = sorted(topK.items(), key=lambda d: d[1],reverse=True)[0:10]
            s = 'Topic %d:\n'%k
            for i in top10:
                s+=str('|-'+i[0].encode('utf-8'))+'\t:\t'+str(i[1]+'\n')
            fp.write(s)
        fp.close()
    def __saveLast(self):
        d = self.__get2DMatrix(self.__M,self.__K)
        # d:元素与文档数相同，每个元素即为主题分布概率
        for m in range(0,self.__M):
            p = [0 for k in range(0,self.__K)]
            for topic in self.__z[m]:
                p[topic] +=1
            s = len(self.__getWordsInFileM(self.__fileData,m))
            if s!=0:
                d[m] = [float(p)/s for p in p]
            else:
                d[m] = [0 for k in range(0,self.__K)]  
        # maxp:返回一个列表，每个元素是文档名以及典型主题集合，其中主题集合为列表，列表中的元素为[主题k,概率]
        maxp = [[] for i in range(0,self.__M)]
        for m in range(0,self.__M):
            n = max(d[m])
            p = {}
            p['name'] = self.__fileData['doc'][m]['file_name']
            p['typical'] = [[i,j] for i,j in enumerate(d[m]) if j==n]
            maxp[m]=p
        # 排序
        maxtopic = {}
        fp = open(os.getcwd()+'\\TypicalItemForTopics','w')
        for k in range(0,self.__K):
            str_ = 'Topic%d'%k
            maxtopic[str_] = {}
            for mp in maxp:
                for cate in mp['typical']:
                    if cate[0]==k:
                        maxtopic[str_][mp['name']]=cate[1]
            maxtopic[str_] = sorted(maxtopic[str_].items(),key=lambda d: d[1],reverse=True)[0:15]
            fp.write(str_+':\n')
            for item in maxtopic[str_]:
                s = '|-'+str(item[0])+'\t:\t'+str(item[1])+'\n'
                fp.write(s)
        fp.close()
