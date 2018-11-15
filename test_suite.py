#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
def main(argv):
    from pprint import pprint
    from numpy import array
    from matplotlib.pyplot import subplots,figure,show,plot,xlabel,ylabel
    from model.model import network,data_feeder,test_suite
    from model.plot import plot_predictions
    from model.landscape import landscape
    from argparse import ArgumentParser
    from numpy import linspace
    from numpy.random import uniform

    p=ArgumentParser()
    p.add_argument("-a","--learning_rate",default=1e-3)
    p.add_argument("-s","--steps",type=int,default=1000)
    p.add_argument("-b","--batch",default=512)
    p.add_argument("-n","--eval_length",default=1024)
    args=p.parse_args()

    #rugged=landscape()
    #x=linspace(-1,2,1000)
    #x=uniform(-1,2,(1024,1))
    #y=rugged.y(x)

    #figure()
    #plot(x,y,',')
    #show()

    suit=test_suite()
    init_vars=tf.group(tf.global_variables_initializer())
    saver=tf.train.Saver()

    with tf.Session() as session: 
        session.run(init_vars)
        session.run(suit.train_iterator.initializer)
        session.run(suit.eval_iterator.initializer)

        train_handle=session.run(suit.train_iterator.string_handle())
        eval_handle=session.run(suit.eval_iterator.string_handle())

        for i in range(args.steps+1):
            l,_=session.run([suit.nn.loss,suit.nn.train],
                    feed_dict={suit.nn.rate: args.learning_rate,
                        suit.handle: train_handle}
                    )
            if i%500 is 0:
                print(i,l)

        saver.save(session,"log/last.ckpt")

        a,b,c=session.run([suit.nn.inputs,suit.nn.outputs,suit.nn.output_layer],
                feed_dict={suit.handle: eval_handle}
                )
    figure()
    plot(a,b)
    plot(a,c)
    show()
    
if __name__=="__main__":
    tf.app.run()
