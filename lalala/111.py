import numpy as np
import tensorflow as tf
a=[[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]
b=np.array(a)
c=tf.transpose(b,perm=[0,2,1])
sess=tf.Session()
#print(a)
#print(sess.run(c))


d=[[2,2,3,1,2,3,4]]
e=np.array(d)
f=tf.one_hot(e,5)
ff=print(sess.run(f))
def reweight(msa1hot, cutoff):
    with tf.name_scope('reweight'):
        id_min = tf.cast(tf.shape(msa1hot)[1], tf.float32) * cutoff#把msa1hot形状张量第2个转为32为浮点类型
        aa=print(sess.run(id_min))
        id_mtx = tf.tensordot(msa1hot, msa1hot, [[1,2], [1,2]])#将`msa1hot'第2,3维度元素和`msa1hot'的第2,3维度元素的点积
        bb=print(sess.run(id_mtx))
        id_mask = id_mtx > id_min
        cc=print(sess.run(id_mask))
        w = 1.0/tf.reduce_sum(tf.cast(id_mask, dtype=tf.float32),-1)#计算tf.cast(id_mask, dtype=tf.float32)在倒数第一个维度上的维度之和
    return w
g=reweight(f,0.8)
xxy=tf.cast(True, dtype=tf.float32)
xxx=tf.cast(False, dtype=tf.float32)
print(sess.run(xxx))
print(sess.run(xxy))
h=sess.run(g)
ddc=[[1,2,3]]
ddt=[[6,5,4],[3,2,1]]

ddd=np.array(ddc)
dddd=np.array(ddt)
ddp=tf.convert_to_tensor(dddd)
dde=tf.convert_to_tensor(ddd)
eee=dde*ddp
print(sess.run(eee))
ee=[1,2,3]
ef=tf.convert_to_tensor(ee)
print(sess.run(ef[:,None]))
print(sess.run(ef[None,:]))