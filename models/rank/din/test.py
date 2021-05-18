import paddle
import numpy as np
inp_np = np.ones([5, 2, 3, 4]).astype('float32')
print("inp_np:",inp_np)
inp_np_tensor = paddle.to_tensor(inp_np)
print("inp_np_tensor:",inp_np_tensor)
flatten = paddle.nn.Flatten(start_axis=0, stop_axis=2)

linear0 = paddle.nn.Linear(in_features=4, out_features=80,)
linear1 = paddle.nn.Linear(in_features=80, out_features=40,)
linear2 = paddle.nn.Linear(in_features=40, out_features=1,)

#reshape0=paddle.reshape(linear0,[5,2,3,80])
#reshape1=paddle.reshape(linear1,[5,2,3,40])
#reshape2=paddle.reshape(linear2,[5,2,3,1])


sig = paddle.nn.Sigmoid()

flatten_res0 = flatten(inp_np_tensor)
linear_res0=linear0(flatten_res0)
#reshape_res0=reshape0(linear_res0)
reshape_res0 = paddle.reshape(linear_res0,[5,2,3,80])
sig_0=sig(reshape_res0)
#sig_0=sig(linear_res0)
print("flatten_res0",flatten_res0)
print("linear_res0",linear_res0)
print("reshape_res0",reshape_res0)
print("sig_0",sig_0)

flatten_res1 = flatten(sig_0)
linear_res1=linear1(flatten_res1)
reshape_res1 = paddle.reshape(linear_res1,[5,2,3,40])
#reshape_res0=reshape0(linear_res0)
sig_1=sig(reshape_res1)
print("flatten_res1",flatten_res1)
print("linear_res1",linear_res1)
print("reshape_res1",reshape_res1)
print("sig_1",sig_1)

flatten_res2 = flatten(sig_1)
linear_res2=linear2(flatten_res2)
reshape_res2 = paddle.reshape(linear_res2,[5,2,3,1])
#reshape_res0=reshape0(linear_res0)
print("flatten_res2",flatten_res2)
print("linear_res2",linear_res2)
print("reshape_res2",reshape_res2)
#print("sig_2",sig_2)
