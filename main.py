#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
def main(argv):
    from pprint import pprint
    from numpy import array
    from matplotlib.pyplot import subplots,figure,show,plot,xlabel,ylabel
    from model.model import network,data_feeder,flow
    from model.plot import plot_predictions
    from argparse import ArgumentParser

    p=ArgumentParser()
    p.add_argument("-a","--learning_rate",default=1e-3)
    p.add_argument("-s","--steps",default=1000)
    p.add_argument("-b","--batch",default=512)
    p.add_argument("-n","--eval_length",default=1024)
    args=p.parse_args()

    fl=flow(batch=args.batch,n=args.eval_length)

    rho=fl.c.get('.rho')
    en=fl.c.get('.en')
    epsilon=fl.c.get('epsilon')

    init_vars=tf.group(tf.global_variables_initializer())
    saver=tf.train.Saver()

    with tf.Session() as session: 
        session.run(init_vars)
        session.run(fl.train_iterator.initializer)
        session.run(fl.eval_iterator.initializer)

        train_handle=session.run(fl.train_iterator.string_handle())
        eval_handle=session.run(fl.eval_iterator.string_handle())

        try:
            saver.restore(session,"log/last.ckpt")
        except tf.errors.NotFoundError:
            pass

        for i in range(args.steps+1):
            l,_=session.run([fl.nn.loss,fl.nn.train],
                    feed_dict={fl.nn.rate: args.learning_rate,
                        fl.handle: train_handle}
                    )
            if i%500 is 0:
                print(i,l)

        saver.save(session,"log/last.ckpt")

        a=session.run([fl.nn.inputs,fl.nn.output_layer],
                feed_dict={fl.handle: eval_handle}
                )

    fig,ax=subplots()
    plot(epsilon,en,alpha=.5)
    plot_predictions(array(a),0)

    fig,ax=subplots()
    plot(epsilon,rho,alpha=.5)
    plot_predictions(array(a),1)

    show()

if __name__=="__main__":
    tf.app.run()
