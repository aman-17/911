import torch
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)   
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

query = inputs[1] #A
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
  print(i, x_i)
  attn_scores_2[i] = torch.dot(x_i, query)
print("Attention scores:", attn_scores_2)

# res = 0.
# for idx, element in enumerate(inputs[0]):
#   res += inputs[0][idx] * query[idx]
# print(res)
# print(torch.dot(inputs[0], query))

# attn_weights_2_tmp = attn_scores_2/attn_scores_2.sum()
# print('Attention Weights: ', attn_weights_2_tmp)
# print('Sum: ', attn_weights_2_tmp.sum())

# def softmax_naive(x):
#   return torch.exp(x) / torch.exp(x).sum(dim=0)

# attn_weights_2_naive = softmax_naive(attn_scores_2)
# print('Attention Weights: ', attn_weights_2_naive)

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)

<<<<<<< HEAD:tokenization/attention_part1.py
query = inputs[1] 
=======
query = inputs[1]
>>>>>>> c0c1a41b7850a63c067069b952323900893537f5:gpt_practise/attention/simplified_self_attention.py
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
  context_vec_2 += attn_weights_2[i] * x_i
  # print(attn_weights_2[i], x_i)
print("Context vector: ", context_vec_2)

# attn_scores = torch.empty(6, 6)
# for i, x_i in enumerate(inputs):
#   for j, x_j in enumerate(inputs):
#     attn_scores[i, j] = torch.dot(x_i, x_j)
# print(attn_scores)

attn_scores = inputs @ inputs.T
# print(attn_scores)

attn_weights = torch.softmax(attn_scores, dim=1)
# print(attn_weights)

all_context_vecs = attn_weights @ inputs
print(all_context_vecs)