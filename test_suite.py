#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
def main(argv):
    """
    Test suite
    """
    from pprint import pprint
    from numpy import array
    from matplotlib.pyplot import subplots,figure,show,plot,xlabel,ylabel,savefig,close,close,ylim,xlim
    from model.model import network,data_feeder,test_suite
    from model.plot import plot_predictions
    from model.landscape import landscape
    from argparse import ArgumentParser
    from numpy import linspace
    from numpy.random import uniform

    p=ArgumentParser()
    p.add_argument("-a","--learning_rate",default=1e-3)
    p.add_argument("-s","--steps",type=int,default=1000)
    p.add_argument("-b","--batch_size",type=int,default=512)
    p.add_argument("-n","--n_eval",type=int,default=1024)
    p.add_argument("-N","--n_samples",type=int,default=1024)
    p.add_argument("-f","--plot_frequency",type=int,default=100)
    args=p.parse_args()

    suit=test_suite(batch_size=args.batch_size,
            n_eval=args.n_eval,
            n_samples=args.n_samples)
    init_vars=tf.group(tf.global_variables_initializer())
    saver=tf.train.Saver()

    with tf.Session() as session: 
        session.run(init_vars)
        session.run(suit.train_iterator.initializer)
        session.run(suit.eval_iterator.initializer)

        train_handle=session.run(suit.train_iterator.string_handle())
        eval_handle=session.run(suit.eval_iterator.string_handle())

        count=0
        for i in range(args.steps+1):
            l,_=session.run([suit.nn.loss,suit.nn.train],
                    feed_dict={suit.nn.rate: args.learning_rate,
                        suit.handle: train_handle}
                    )
            if i%args.plot_frequency is 0:
                print(i,l)
                a,b,c=session.run([suit.nn.inputs,suit.nn.outputs,suit.nn.output_layer],
                        feed_dict={suit.handle: eval_handle}
                        )

                fig,ax=subplots()
                ax.plot(a,b,"k-")
                ax.plot(a,c)
                ax.set_ylim(-4.5,-0.5)
                ax.set_title("step {:d}".format(i),fontsize=10)

                savefig("log/{:03d}.png".format(count))
                close(fig)

                count+=1

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
