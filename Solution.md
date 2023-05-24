
## Question: Find a way to use GPT models without sending the data to openai.

To solve this problem we may have 2 approaches:
1. Where we do not have enough data to perform training.
2. Where we have sufficient data for model training.

***Proposed architectural diagram***
![openai_main.png](/openai_main.png)


================== NOTES =======================
### General Introduction about GPT models
In 2022 GPT-3.5 was released with about 175B trainable params, it can be accessed commercially using ChatGPT or professionally using openai's GPT's API, where you have to pay to gain access to run your data on an LLM and get your data back. In GPT-3.5 we access the LLM models through the prompt (where we can type something in).

GPT-4 was released in 2023, which is able to handle text, images and code. The number of params this models has, is not yet disclosed (top secret). GPT-4 model is a paid model which can be accessed using ChatGPT PLUS or its API (cost GPT-4 > GPT-3.5) for professionals.

### Google's approach to creating models for a specific task
In 2022, google came up with a model based off of the transformer model called T5 (text-to-text-transfer-transformer). This model came about as a result of Google's attempt to generate a special high performance system to handle a specific task. The model had two variations. One with 3B and the other with 11B trainable parameters, which was impressive at the time.

Google discovered GPT models were trained on internet data. However, most companies will like to use their custom special/secret data i.e: (company data, medical records, private photos etc).

Google found out that if we take out some piece of special data to be used for a specific task say: translation task, and we want to change from inferential language to another language say a language spoken in the south pacific, there isn't enough data on the internet to this.  Google proposed that if we fine-tuned our training on the T5 model given that GPT-3 and GPT-4 are already pre-trained models (digested all the internet data) it will be able to perform specific tasks.

As a result, if as a user, you're not satisfied with the pre-training of the already existing GPT models, you can fine-tune your system for one particular task which is how google came out with the Flan-T5 model (Fine-tuned language).  

**NB: This is how google cloud architecture came about, It was not fine-tuned on one task but on multiple tasks.**

This was how google made use of it's company secret data to gain access to the architectural intelligence of an LLM of a transformer. This made people use fine-tuning with their private data on their dedicated hardware to solve specific tasks.

```
Lets assume for GPT-3.5 we need about 1000GPU's on a microsoft super cluster to train a GPT-3.5 model, and for GPT-4 we need a hardware configuration of about 1K to 10K GPU's to train this model. Even if we had the access to GPT-4 through ChatGPT Plus or it's API we are not given any information about what happens in this model. It is a black box. Also, GPT-4 as of right now cannot be fined-tuned for a specific task because it has been overly pre-trained.
```

### Solution to GPT-4's inability to perform Fine Tuning

With the existence of fine-tuning and the advancement of models, it was discovered that you cannot only prompt engineer your way to ChatGPT (GPT-3.5) but we can make use of In-Context learning (ICL) for GPT-4.

ICL is an advanced prompt engineering methodology, were a piece of your private data is sliced/broken into tiny chunks and we put in a piece of information and we ask something, we then put in the next piece and so on. This means we can feed in small small pieces of information and then ask the system for a result.

Since the learnable params of the GPT-4 model does not change because it is top-secret, we are not able to do fine-tuning. This is because in fine-tuning the learnable params as we saw in Flan-T5 model were learning on the new training data.

The beauty of In-Context learning is that you can chain different methodologies together on the small chunks of your data. This was the approach used to create to create a learnable surface for on GPT-4 model. Nevertheless, we still aren't able to learn to train the system on our data.

## LLama Model - **delete?? (not sure)**
As a result of this Meta created the LLama model for research purposes only. LLama model is available in 4 sizes: small, medium, large, XL with 7B, 13B, 33B and 65B trainable params respectively.

Looking at the size of small LLama model, it has only 7B params which is really really small compared to GPT-4. However, the advantage is that 7B params can be handled by an infrastructure at home or in your company. However, only big companies like Microsoft can afford thousands of GPU's to train GPT models.

Apart from Flan-T5 which was open sourced by google, the  LLama model also  came into play.

When the internal details of the LLama model was leaked. It was discovered that it needed a lot of training data (about 50K), but if you had data that was intelligently configured which had a lot of inherent information, you could use this intelligent data and feed it to the smallest of the LLama model which is what majority of the AI community has access to.

To run the data on the smallest LLama model we need only 8 GPU's. 

### Solution to the use case where there's insufficient data to train a model. 
People discovered that if you do not have the company/secret data, but you have a small set of data that has a particular form. E.g: you want to write a poem and a human drafts the initial poem. We can feed this initial (human drafted) poem to GPT-4 and using its payable API ask it to generate about 50K multiple versions of this same poem which is called **self-instruct data** making variations to different aspects of the poem such as: verbs, objects, time, person, outcome etc.

Now for our training if we used 50K training dataset, which is the right amount need for small LLama model, this model can be fine-tuned for a particular task. This was how the ALPACA model was created by Stanford University.

This is the process how you now access.... you have some data (secret) give an instruction to GPT-4 to create thousands of similar instructions, because we need huge training data to feed to the smallest LLama model to develop a system intelligence of an LLM and that was how ALPACA was created.

```
Process:
Sample of initial data -> generate syntethic data from GPT-4 -> feed to smallest LLama model.
```

**NB: This approach brought about the existence of many of the models we see today, such as: ViCUNA, GPT-4ALL, e.t.c***


### Approaches used in Fine Tuning a model
To fine tune a LLama model if you don't have enough money for an expensive infrastructure from microsoft, we could used 2 approaches:
1. Parameter Efficient Fine Tuning -PEFT (LORA) - Here since our technological infrastructure is so reduced, we Freeze the LL model and work with trainable weights (<1% of all weights).
2.  Classical Fine Tuning (FT) - Used when you have the money to rent the infrastructure (supercomputers) needed. Here all the weights are trainable (100% of weight tensors: 8x H100).
3.  Instruction Fine Tuning - (Complex Data): Used when there's sufficient data, and our task is to obtain patterns from this complex data. 

```
NB: Alapaca used Classical Fine Tuning after generating self-instruct data using GPT-4
```

### Solution to the use case where there's sufficient data to train a model
If you have enough data in your company you don't need go to for the synthetic data generation approach. You can train your model using the .Instruction Fine Tuning approach, designed to learn complex patterns from our dataset.

Assuming you have your huge company secret data say: Quantum chemistry, Quantum Physics and Pharmacology etc, that we do not want to put on the internet, and our goal is to find new medications or  whatever you're researching for, you can combine all the data from the 3 fields together e.g 1k research papers on all three fields. To find common patterns which no human has discovered, we feed all these data and use the Instruction Fine Tuning which approach used for complex data set to train either the medium, large or XL LLama model.

We don't need the huge size of internet data which was used to train the GPT-4 model, since we only want to train on our own already available dataset.

Lastly, Alpaca paid $600 for computer infrastucture, to generate the data, the paid mircosoft $500 and then to run it on a supercomputer the paid only $100. Therefore, the generation of data in value of high quality data, the data was 5 times more valueable to generate than to run the operation on the supercomputer.

```
We can substitute LLama (ALPACA) for Flan-T5 or DOLLY 2 since the LLama model is only avialable for research purposes and its weights are secret.
```

### Brief Intro on DOLLY 2
DOLLY 2 is the first open source instruction-following Pythia LLM, fine-tuned on DataBricks human-generated instruction dataset. It has 12B parameters trained on 15k in-house data and it is also available for research and commercial use.
