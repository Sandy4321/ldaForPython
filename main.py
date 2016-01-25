# -*- coding:utf-8 -*- 
import lda
import simplejson as sj
import os
import time


ISOTIMEFORMAT = '%Y-%m-%d %X'

print 'start at:'+time.strftime( ISOTIMEFORMAT, time.localtime() )



p = lda.LdaModel()
p.readFile('YOUR DATA DIRECTION')
print 'file reading finish...'
p.DocumentInitiation()
print 'initiation finish...'
p.lda_inference()
print 'lda inference...'



print 'end at:'+time.strftime( ISOTIMEFORMAT, time.localtime() )
