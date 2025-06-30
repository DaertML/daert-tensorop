# daert-tensorop
Open source implementation of the QiMeng-TensorOp: https://arxiv.org/pdf/2505.06302

![alt text](image-10.webp "A llama doing gpus go brrr")

# Introduction
The main idea goes beyond the daertGEMM repo; in this case, any tensor operation is evolved into an optimized kernel.

As a final improvement, MCTS runs with the first generated kernel from the LLM of choice, and evolved it my finetuning the different hyperparameters of the kernel and hardware optimization hints.

# Run it
It is expected that you add PDF files with the manuals that you want to include about the different hardware platforms to write kernels for. Also, it is expected to have a local Qdrant vector store.

Execute the embed.py executable to feed the vector store, then langgraph_workflow.py

The MCTS part is WIP at the moment.

# Future plans
The main objective is to replicate the QiMeng results and go beyond.
