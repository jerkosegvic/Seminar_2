## A survey on Large Language Model interpretability methods
This seminar explores methods for interpreting LLM’s hidden states. Since all transformer
blocks in LLMs are the same in terms of architecture (they have different parameters), the
question is what features are detected in each layer. One way to approach this problem is to
see if it’s possible to extract information about the next token prediction from every hidden
state and see how the performance changes from one state to another. We evaluated the
model with a method called logit lens, which does not require any additional training, and
tuned lens, which adds a linear layer on top of every transformer block and learns projection
to the final hidden state. We measure the performance of every layer with KL − divergence,
the probability of the correct token, and the rank of the correct token. Furthermore, we
discuss the meaning of the results and suggest further research.
