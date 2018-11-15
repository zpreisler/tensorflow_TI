import tensorflow as tf
class network(object):
    def __init__(self,inputs,outputs,name='network'):
        print('Initialize %s'%name)

        self.rate=tf.placeholder(tf.float64)

        self.inputs=inputs
        self.outputs=outputs
        self._loss=None
        self._optimizer=None
        self._train=None

        self.build_graph(inputs,output_dim=self.outputs.shape[-1],name=name)

        self.loss=self.__loss
        self.train=self.__train

    def build_graph(self,inputs,output_dim,name):
        with tf.variable_scope(name):
            self.dense_1=tf.layers.dense(inputs=inputs,
                    units=16,
                    activation=tf.nn.tanh,
                    name='d1')

            self.dense_2=tf.layers.dense(inputs=self.dense_1,
                    units=8,
                    activation=tf.nn.tanh,
                    name='d2')

            self.dense_3=tf.layers.dense(inputs=self.dense_2,
                    units=output_dim,
                    name='output_layer')

            self.output_layer=self.dense_3

    @property
    def __loss(self):
        if not self._loss:
            self._loss=tf.reduce_mean(
                    tf.nn.l2_loss(
                        self.output_layer-self.outputs
                        )
                    )
        return self._loss

    @property
    def optimizer(self):
        if not self._optimizer:
            self._optimizer=tf.train.AdamOptimizer(learning_rate=self.rate)
        return self._optimizer

    @property
    def __train(self):
        if not self._train:
            self._train=self.optimizer.minimize(self.loss)
        return self._train

class flow(object):
    def __init__(self,name='flow',batch=512,n=1024):
        from .data_pipeline import data_pipeline,data_pipeline_handle

        self.c=data_feeder(files='eos/fluid*.conf',
                add_data=['.en','.rho'])

        self.data=self.c.feed(['epsilon','pressure','.en','.rho'])
        self.data_all=self.c.feed_data(['epsilon','pressure','.en','.rho'])

        inputs=self.data[:,:2] #epsilon,pressure
        outputs=self.data[:,2:] #en,rho

        self.handle=tf.placeholder(tf.string,shape=[])

        next_element,self.train_iterator,self.eval_iterator=data_pipeline_handle(self.handle,inputs=inputs,
                outputs=outputs,batch_size=batch,n=n)

        """
        Network
        """
        self.nn=network(next_element['inputs'],
                next_element['outputs'],
                name='Kagami')

class test_suite(object):
    def __init__(self,name='test_suite',batch=512,n_samples=2048,n=1024):
        from .data_pipeline import data_pipeline,data_pipeline_handle
        from .landscape import landscape
        from numpy import linspace
        from numpy.random import uniform

        self.rugged=landscape()
        
        inputs=uniform(-1,2,(n_samples,2))
        outputs=self.rugged.y(inputs)

        print(inputs.shape)
        print(outputs.shape)

        self.handle=tf.placeholder(tf.string,shape=[])

        next_element,self.train_iterator,self.eval_iterator=data_pipeline_handle(self.handle,inputs=inputs,
                outputs=outputs,batch_size=batch,n=n)

        self._inputs=next_element['inputs']
        self._outputs=next_element['outputs']
        """
        Network
        """
        self.nn=network(next_element['inputs'],
                next_element['outputs'],
                name='Asuna')

from myutils import configuration,data
class data_feeder(configuration):
    def __init__(self,files,add_data=[],delimiter=[':']):
        from glob import glob
        print("Data Feeder")
        files=glob(files)
        configuration.__init__(self,files,delimiter=delimiter,add_data=add_data)

        self.dsort(key=lambda x: float(*x['epsilon']))

    def __is_data__(self,name):
        return any([isinstance(x[name],data) for x in self.dconf])

    def get(self,name):
        if self.__is_data__(name):
            return [x[name].data.mean() for x in self.dconf]
        else:
            return [float(*x[name]) for x in self.dconf]

    def feed(self,names=[]):
        from numpy import array
        d=[]
        for x in self.dconf:
            v=[x[name] for name in names]
            dd=[]
            for t in v:
                if isinstance(t,data):
                    dd+=[t.data.mean()]
                else:
                    dd+=[float(*t)]
            d+=[dd]
        return array(d)

    def feed_data(self,names=[]):
        from numpy import array,append,vstack,concatenate,ones,hstack,full
        d=[]
        for x in self.dconf:
            t=[]
            w=[x[name].data for name in names if isinstance(x[name],data) is True]
            m=min([len(y) for y in w])

            for name in names:
                if isinstance(x[name],data) is False:
                    v=float(*x[name])
                    t+=[full(m,v)]

            w=array(t+w).transpose()
            d+=[w]

        d=concatenate(d)
        return d
