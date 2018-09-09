# improving-performance
model.fit(X_train_pad, y_train,
          batch_size=32,
          epochs=2,
          validation_split=0.3)
  
  
Train on 17500 samples, validate on 7500 samples
Epoch 1/2
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
~\Miniconda2\envs\ztdl\lib\site-packages\tensorflow\python\client\session.py in _do_call(self, fn, *args)
   1038     try:
-> 1039       return fn(*args)
   1040     except errors.OpError as e:

~\Miniconda2\envs\ztdl\lib\site-packages\tensorflow\python\client\session.py in _run_fn(session, feed_dict, fetch_list, target_list, options, run_metadata)
   1020                                  feed_dict, fetch_list, target_list,
-> 1021                                  status, run_metadata)
   1022 

~\Miniconda2\envs\ztdl\lib\contextlib.py in __exit__(self, type, value, traceback)
     65             try:
---> 66                 next(self.gen)
     67             except StopIteration:

~\Miniconda2\envs\ztdl\lib\site-packages\tensorflow\python\framework\errors_impl.py in raise_exception_on_not_ok_status()
    465           compat.as_text(pywrap_tensorflow.TF_Message(status)),
--> 466           pywrap_tensorflow.TF_GetCode(status))
    467   finally:

InvalidArgumentError: indices[0,13] = 48983 is not in [0, 20000)
	 [[Node: embedding_3/Gather = Gather[Tindices=DT_INT32, Tparams=DT_FLOAT, validate_indices=true, _device="/job:localhost/replica:0/task:0/cpu:0"](embedding_3/embeddings/read, _recv_embedding_3_input_0)]]

During handling of the above exception, another exception occurred:

InvalidArgumentError                      Traceback (most recent call last)
<ipython-input-70-786c7bc2a397> in <module>()
      2           batch_size=32,
      3           epochs=2,
----> 4           validation_split=0.3)

~\Miniconda2\envs\ztdl\lib\site-packages\keras\models.py in fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, **kwargs)
    854                               class_weight=class_weight,
    855                               sample_weight=sample_weight,
--> 856                               initial_epoch=initial_epoch)
    857 
    858     def evaluate(self, x, y, batch_size=32, verbose=1,

~\Miniconda2\envs\ztdl\lib\site-packages\keras\engine\training.py in fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, **kwargs)
   1496                               val_f=val_f, val_ins=val_ins, shuffle=shuffle,
   1497                               callback_metrics=callback_metrics,
-> 1498                               initial_epoch=initial_epoch)
   1499 
   1500     def evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None):

~\Miniconda2\envs\ztdl\lib\site-packages\keras\engine\training.py in _fit_loop(self, f, ins, out_labels, batch_size, epochs, verbose, callbacks, val_f, val_ins, shuffle, callback_metrics, initial_epoch)
   1150                 batch_logs['size'] = len(batch_ids)
   1151                 callbacks.on_batch_begin(batch_index, batch_logs)
-> 1152                 outs = f(ins_batch)
   1153                 if not isinstance(outs, list):
   1154                     outs = [outs]

~\Miniconda2\envs\ztdl\lib\site-packages\keras\backend\tensorflow_backend.py in __call__(self, inputs)
   2227         session = get_session()
   2228         updated = session.run(self.outputs + [self.updates_op],
-> 2229                               feed_dict=feed_dict)
   2230         return updated[:len(self.outputs)]
   2231 

~\Miniconda2\envs\ztdl\lib\site-packages\tensorflow\python\client\session.py in run(self, fetches, feed_dict, options, run_metadata)
    776     try:
    777       result = self._run(None, fetches, feed_dict, options_ptr,
--> 778                          run_metadata_ptr)
    779       if run_metadata:
    780         proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)

~\Miniconda2\envs\ztdl\lib\site-packages\tensorflow\python\client\session.py in _run(self, handle, fetches, feed_dict, options, run_metadata)
    980     if final_fetches or final_targets:
    981       results = self._do_run(handle, final_targets, final_fetches,
--> 982                              feed_dict_string, options, run_metadata)
    983     else:
    984       results = []

~\Miniconda2\envs\ztdl\lib\site-packages\tensorflow\python\client\session.py in _do_run(self, handle, target_list, fetch_list, feed_dict, options, run_metadata)
   1030     if handle is None:
   1031       return self._do_call(_run_fn, self._session, feed_dict, fetch_list,
-> 1032                            target_list, options, run_metadata)
   1033     else:
   1034       return self._do_call(_prun_fn, self._session, handle, feed_dict,

~\Miniconda2\envs\ztdl\lib\site-packages\tensorflow\python\client\session.py in _do_call(self, fn, *args)
   1050         except KeyError:
   1051           pass
-> 1052       raise type(e)(node_def, op, message)
   1053 
   1054   def _extend_graph(self):

InvalidArgumentError: indices[0,13] = 48983 is not in [0, 20000)
	 [[Node: embedding_3/Gather = Gather[Tindices=DT_INT32, Tparams=DT_FLOAT, validate_indices=true, _device="/job:localhost/replica:0/task:0/cpu:0"](embedding_3/embeddings/read, _recv_embedding_3_input_0)]]

Caused by op 'embedding_3/Gather', defined at:
  File "C:\Users\Administrator\Miniconda2\envs\ztdl\lib\runpy.py", line 193, in _run_module_as_main
    "__main__", mod_spec)
  File "C:\Users\Administrator\Miniconda2\envs\ztdl\lib\runpy.py", line 85, in _run_code
    exec(code, run_globals)
  File "C:\Users\Administrator\Miniconda2\envs\ztdl\lib\site-packages\ipykernel_launcher.py", line 16, in <module>
    app.launch_new_instance()
  File "C:\Users\Administrator\Miniconda2\envs\ztdl\lib\site-packages\traitlets\config\application.py", line 658, in launch_instance
    app.start()
  File "C:\Users\Administrator\Miniconda2\envs\ztdl\lib\site-packages\ipykernel\kernelapp.py", line 486, in start
    self.io_loop.start()
  File "C:\Users\Administrator\Miniconda2\envs\ztdl\lib\site-packages\tornado\platform\asyncio.py", line 132, in start
    self.asyncio_loop.run_forever()
  File "C:\Users\Administrator\Miniconda2\envs\ztdl\lib\asyncio\base_events.py", line 421, in run_forever
    self._run_once()
  File "C:\Users\Administrator\Miniconda2\envs\ztdl\lib\asyncio\base_events.py", line 1425, in _run_once
    handle._run()
  File "C:\Users\Administrator\Miniconda2\envs\ztdl\lib\asyncio\events.py", line 127, in _run
    self._callback(*self._args)
  File "C:\Users\Administrator\Miniconda2\envs\ztdl\lib\site-packages\tornado\platform\asyncio.py", line 122, in _handle_events
    handler_func(fileobj, events)
  File "C:\Users\Administrator\Miniconda2\envs\ztdl\lib\site-packages\tornado\stack_context.py", line 300, in null_wrapper
    return fn(*args, **kwargs)
  File "C:\Users\Administrator\Miniconda2\envs\ztdl\lib\site-packages\zmq\eventloop\zmqstream.py", line 450, in _handle_events
    self._handle_recv()
  File "C:\Users\Administrator\Miniconda2\envs\ztdl\lib\site-packages\zmq\eventloop\zmqstream.py", line 480, in _handle_recv
    self._run_callback(callback, msg)
  File "C:\Users\Administrator\Miniconda2\envs\ztdl\lib\site-packages\zmq\eventloop\zmqstream.py", line 432, in _run_callback
    callback(*args, **kwargs)
  File "C:\Users\Administrator\Miniconda2\envs\ztdl\lib\site-packages\tornado\stack_context.py", line 300, in null_wrapper
    return fn(*args, **kwargs)
  File "C:\Users\Administrator\Miniconda2\envs\ztdl\lib\site-packages\ipykernel\kernelbase.py", line 283, in dispatcher
    return self.dispatch_shell(stream, msg)
  File "C:\Users\Administrator\Miniconda2\envs\ztdl\lib\site-packages\ipykernel\kernelbase.py", line 233, in dispatch_shell
    handler(stream, idents, msg)
  File "C:\Users\Administrator\Miniconda2\envs\ztdl\lib\site-packages\ipykernel\kernelbase.py", line 399, in execute_request
    user_expressions, allow_stdin)
  File "C:\Users\Administrator\Miniconda2\envs\ztdl\lib\site-packages\ipykernel\ipkernel.py", line 208, in do_execute
    res = shell.run_cell(code, store_history=store_history, silent=silent)
  File "C:\Users\Administrator\Miniconda2\envs\ztdl\lib\site-packages\ipykernel\zmqshell.py", line 537, in run_cell
    return super(ZMQInteractiveShell, self).run_cell(*args, **kwargs)
  File "C:\Users\Administrator\Miniconda2\envs\ztdl\lib\site-packages\IPython\core\interactiveshell.py", line 2662, in run_cell
    raw_cell, store_history, silent, shell_futures)
  File "C:\Users\Administrator\Miniconda2\envs\ztdl\lib\site-packages\IPython\core\interactiveshell.py", line 2785, in _run_cell
    interactivity=interactivity, compiler=compiler, result=result)
  File "C:\Users\Administrator\Miniconda2\envs\ztdl\lib\site-packages\IPython\core\interactiveshell.py", line 2901, in run_ast_nodes
    if self.run_code(code, result):
  File "C:\Users\Administrator\Miniconda2\envs\ztdl\lib\site-packages\IPython\core\interactiveshell.py", line 2961, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
  File "<ipython-input-67-c4f56c46848e>", line 2, in <module>
    model.add(Embedding(max_features, 128))
  File "C:\Users\Administrator\Miniconda2\envs\ztdl\lib\site-packages\keras\models.py", line 433, in add
    layer(x)
  File "C:\Users\Administrator\Miniconda2\envs\ztdl\lib\site-packages\keras\engine\topology.py", line 585, in __call__
    output = self.call(inputs, **kwargs)
  File "C:\Users\Administrator\Miniconda2\envs\ztdl\lib\site-packages\keras\layers\embeddings.py", line 120, in call
    out = K.gather(self.embeddings, inputs)
  File "C:\Users\Administrator\Miniconda2\envs\ztdl\lib\site-packages\keras\backend\tensorflow_backend.py", line 1072, in gather
    return tf.gather(reference, indices)
  File "C:\Users\Administrator\Miniconda2\envs\ztdl\lib\site-packages\tensorflow\python\ops\gen_array_ops.py", line 1207, in gather
    validate_indices=validate_indices, name=name)
  File "C:\Users\Administrator\Miniconda2\envs\ztdl\lib\site-packages\tensorflow\python\framework\op_def_library.py", line 768, in apply_op
    op_def=op_def)
  File "C:\Users\Administrator\Miniconda2\envs\ztdl\lib\site-packages\tensorflow\python\framework\ops.py", line 2336, in create_op
    original_op=self._default_original_op, op_def=op_def)
  File "C:\Users\Administrator\Miniconda2\envs\ztdl\lib\site-packages\tensorflow\python\framework\ops.py", line 1228, in __init__
    self._traceback = _extract_stack()

InvalidArgumentError (see above for traceback): indices[0,13] = 48983 is not in [0, 20000)
	 [[Node: embedding_3/Gather = Gather[Tindices=DT_INT32, Tparams=DT_FLOAT, validate_indices=true, _device="/job:localhost/replica:0/task:0/cpu:0"](embedding_3/embeddings/read, _recv_embedding_3_input_0)]]

  
