import paddle
import paddle.nn.functional as F

logit = paddle.to_tensor([5.0, 4.0, 2.0, 1.0, 3.0], dtype="float32")
label = paddle.to_tensor([1.0, 0.0, 1.0, 2.0, 0.0], dtype="float32")

sig_label=F.sigmoid(logit)
print("sig_label",sig_label)
print("label",label)
sig_label_0=sig_label.reshape([-1,1])
label_0=label.reshape([-1,1])
print("sig_label_0",sig_label_0)
print("label_0",label_0)

log_cost = F.log_loss(input=sig_label_0, label=label_0)
print("log_cost",log_cost)
log_cost_mean=paddle.mean(log_cost)
print("log_cost_mean",log_cost_mean)

output = paddle.nn.functional.binary_cross_entropy_with_logits(logit, label)
print(output)  # [0.45618808]
