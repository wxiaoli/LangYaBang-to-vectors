# coding=utf-8

from collections import Counter
import simplejson as sjson
import chardet #detect encoding of file
import jieba #分词
import jieba.analyse
import re
import os
import gensim, logging
from gensim.models import Word2Vec

###########################sub funs#################################################
def getData(fname):
    fobj=open(fname,'r')
    docu=fobj.read()
    fobj.close()
    return docu

def removePunc(docu):
    punc=u"[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）：”“～＾０－《》‘’\-]+"
    docu=docu.decode(chardet.detect(docu)['encoding'])
    docu_dpunc=re.sub(punc,'',docu)
    return docu_dpunc

def cutByJieba(document,fnameStr,fnameWordFre):
    #把下面人名添加到词典中（毕竟古人名不常见）
    jieba.add_word(u'琅琊')
    jieba.add_word(u'萧景睿')
    jieba.add_word(u'云飘蓼')
    jieba.add_word(u'梅长苏')
    jieba.add_word(u'谢弼')
    jieba.add_word(u'言豫津')
    jieba.add_word(u'萧景琰')
    jieba.add_word(u'霓凰')
    jieba.add_word(u'飞流') 
    
    #切
    docu_cut=' '.join(jieba.cut(document))
    
    #保存切分结果到文件fnameStr
    fobj1=open(fnameStr,'w')
    fobj1.write(docu_cut.encode('utf-8'))    
    fobj1.close()
    
    #保存词频到文件fnameWordFre
    fobj2=open(fnameWordFre,'w')
    word_fre=Counter(docu_cut.split())
    ##print u'共有%d个词'%len(word_fre) ##39297
    sjson.dump(word_fre,fobj2)
    fobj2.close()
    
    #输出词频top100中多于一个字的词
    for a in word_fre.most_common(100):
        if len(a[0])>3:
            print(a[0] + '\t' + str(a[1]))

def trainBygensim(docuFile,vecFile):    
    #logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    data_path = docuFile
    class MySentences(object):
        def __init__(self, dirname):
            self.dirname = dirname
     
        def __iter__(self):
            for fname in os.listdir(self.dirname):
                for line in open(os.path.join(self.dirname, fname)): #, encoding='utf8'
                    yield line.split()
    sentences = MySentences(data_path) # a memory-friendly iterator
    model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4) #size=100：词向量长度设为100
    model.save(vecFile)
    
    #task1:找出某一个词向量最相近的topn个词
    for key in model.wv.similar_by_word('梅长苏', topn =20):
        print key[0],key[1]  
    
    #task2:看两个词向量的相近程度
    print 'distance(梅长苏,霓凰):%f'%model.wv.similarity('梅长苏','霓凰')
    print 'distance(梅长苏,飞流):%f'%model.wv.similarity('梅长苏','飞流')


###########################main()#################################################
if __name__ == '__main__':
    docu0=getData('./lyb.txt')
    cutByJieba(docu0,'./lyb_segment/lyb_segment.txt','./lyb_wf.txt') 
    trainBygensim('./lyb_segment','./lyb_vec1')
    
    #test
    model=Word2Vec.load('./lyb_vec1')    
    print 'distance(梅长苏,霓凰):%f'%model.wv.similarity('梅长苏','霓凰')
    print 'distance(梅长苏,飞流):%f'%model.wv.similarity('梅长苏','飞流')
    