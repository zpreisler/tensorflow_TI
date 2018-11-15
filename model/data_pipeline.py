def data_pipeline(inputs=None,outputs=None,batch=2048,n=1024):
        import tensorflow as tf
        from numpy import linspace,zeros,array
        """
        train dataset
        """
        length=len(inputs)
        if batch>length:
            batch=length

        dataset=tf.data.Dataset.from_tensor_slices(
                {'inputs': inputs,
                    'outputs': outputs}
                )
        train_dataset=dataset.repeat().shuffle(length).batch(batch)

        iterator=tf.data.Iterator.from_structure(
                train_dataset.output_types,
                train_dataset.output_shapes)
        next_element=iterator.get_next()
        init_train_op=iterator.make_initializer(train_dataset)

        """
        eval dataset
        """
        x=[]
        for k in inputs.transpose():
            x+=[linspace(k.min(),k.max(),n)]
        x=array(x).transpose()
        z=zeros((n,outputs.shape[-1]))

        dataset=tf.data.Dataset.from_tensor_slices( 
                {'inputs': x,
                    'outputs': z}
                )
        eval_dataset=dataset.batch(n)

        init_eval_op=iterator.make_initializer(eval_dataset)

        return next_element,init_train_op,init_eval_op

def data_pipeline_handle(handle,inputs=None,outputs=None,batch=2048,n=1024):
        import tensorflow as tf
        from numpy import linspace,zeros,array
        """
        train dataset
        """
        length=len(inputs)

        dataset=tf.data.Dataset.from_tensor_slices(
                {'inputs': inputs,
                    'outputs': outputs}
                )
        train_dataset=dataset.repeat().shuffle(length).batch(batch)

        iterator=tf.data.Iterator.from_string_handle(
                handle,
                train_dataset.output_types,
                train_dataset.output_shapes)

        next_element=iterator.get_next()
        training_iterator=train_dataset.make_initializable_iterator()

        """
        eval dataset
        """
        x=[]
        for k in inputs.transpose():
            x+=[linspace(k.min(),k.max(),n)]
        x=array(x).transpose()
        z=zeros((n,outputs.shape[-1]))

        dataset=tf.data.Dataset.from_tensor_slices( 
                {'inputs': x,
                    'outputs': z}
                )
        eval_dataset=dataset.repeat().batch(n)
        eval_iterator=eval_dataset.make_initializable_iterator()

        return next_element,training_iterator,eval_iterator
