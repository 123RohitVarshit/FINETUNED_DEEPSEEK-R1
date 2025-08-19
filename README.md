# Medical Expert AI: Fine-Tuning DeepSeek-LLM for Advanced Clinical Reasoning 🩺

This project takes a powerful, general-purpose AI and transforms it into a specialized medical expert. By training it on a unique dataset of medical questions and reasoning, we enhance its ability to provide accurate, logical, and well-thought-out answers in the medical domain.

The secret sauce is using **Unsloth**, a library that makes this advanced fine-tuning process incredibly fast and memory-efficient, allowing you to run it on a single consumer-grade GPU.

## The Big Idea 💡

We're taking a brilliant but general AI, the `DeepSeek-R1-Llama-8B` model, and giving it a focused education in medicine. We do this by showing it examples of complex medical questions and the step-by-step reasoning an expert would use to arrive at an answer. After this training, the model is not just repeating facts—it's learning *how to think* like a medical professional.

## What's Under the Hood? 🛠️ (The Tech Explained)

This project stands on the shoulders of some amazing technologies. Here’s a quick rundown of what each one does:

*   🧠 **DeepSeek-LLM**: Our base model. Think of it as a highly intelligent student who is a fast learner. It's already great at reasoning, making it the perfect candidate for medical specialization.

*   ⚡ **Unsloth**: The magic wand for speed and efficiency. It rewrites parts of the model's code in the background, making the training process up to **2x faster** and using **70% less GPU memory**. It’s what makes fine-tuning an 8-billion-parameter model on your own machine possible.

*   🎓 **Fine-Tuning (SFT)**: The process of teaching our AI a new skill. We take the pre-trained DeepSeek model and "fine-tune" it on our specific medical dataset. It's like sending a brilliant graduate to medical school.

*   ✍️ **LoRA (Low-Rank Adaptation)**: The smart way to fine-tune. Instead of re-training the entire massive model (which would be slow and require immense resources), LoRA freezes the original model and adds tiny, trainable "adapter" layers. It's like adding small, clever sticky notes to a textbook instead of rewriting the whole book. This is a form of **PEFT** (Parameter-Efficient Fine-Tuning).

*   🗜️ **4-bit Quantization (QLoRA)**: A clever trick to shrink the model's memory footprint. It reduces the precision of the model's numbers (its "weights"), making it much lighter to handle without losing much performance. This is a key part of how we fit a huge model into a small GPU.

*   🤔 **Chain-of-Thought Prompting**: A special way of formatting our questions. Instead of just asking for an answer, we ask the model to first "think step-by-step" and outline its reasoning *before* giving the final response. This leads to more logical and trustworthy answers.

*   🤗 **The Hugging Face Ecosystem**: The ultimate toolkit for AI.
    *   `transformers`: For loading and interacting with the model.
    *   `datasets`: For easily downloading and preparing our training data.
    *   `trl`: For orchestrating the fine-tuning process with LoRA.

*   📊 **Weights & Biases (`wandb`)**: Our digital lab notebook. It automatically logs all the important metrics during training (like how fast the model is learning), so we can track our experiments and see what works best.
## The Guided Tour (Jupyter Notebook)
This is the best way to start. The notebook lets you run the code cell-by-cell, inspect outputs, and truly understand the process.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github.com/123RohitVarshit/FINETUNED_DEEPSEEK-R1/blob/main/FINETUNE_DEEPSEEK%20(1).ipynb)

Your Journey in Colab:
1. **Open in Colab**: Click the badge above to launch the notebook.
2. **Activate the GPU**: In the menu, navigate to Runtime -> Change runtime type and select a GPU accelerator (like the T4).
3. **Store Your Secrets**: On the left sidebar, click the key icon (Secrets) and add your API keys. This keeps them safe and secure.

   * Name: Hugging_Face_Token, Value: your_hf_..._token
   * Name: wnb, Value: your_wandb_..._key
      
5. **Run the Show**: Execute the cells from top to bottom and watch your medical AI come to life!

## How It Works: A Step-by-Step Journey 🚶‍♂️

The Python script guides the model through its medical school training:

1.  **Setting the Stage**: We install all the necessary tools and log into Hugging Face and Weights & Biases to get everything ready.

2.  **Loading Our Star Pupil**: We load the `DeepSeek-R1` model using Unsloth. It’s immediately optimized, quantized to 4-bit, and ready for efficient training.

3.  **The "Before" Snapshot**: We test the model on a medical question *before* any training to get a baseline. This shows us its general knowledge.

4.  **Preparing the Textbooks**: We load our medical dataset and format each entry into our special "chain-of-thought" prompt structure. This is the study material for our model.

5.  **The Smart Training Setup**: We apply LoRA to the model. This tells the trainer to only update the small, efficient adapter layers, saving a huge amount of time and memory.

6.  **Time to Train!**: We kick off the training process using the `SFTTrainer`. The model reads our formatted medical data and learns the patterns of clinical reasoning. We can watch its progress live in our Weights & Biases dashboard.

7.  **The "After" Snapshot**: Once training is complete, we ask the model the *same* medical question from the beginning. We can now see a dramatic improvement in the quality, structure, and reasoning of its response.

## Ready to Try It Yourself? 🚀

### Prerequisites

*   A Python environment (like Google Colab).
*   An NVIDIA GPU is highly recommended to make this run quickly.
*   A free [Hugging Face](https://huggingface.co/join) account and an API token.
*   A free [Weights & Biases](https://wandb.ai/site) account and API key.

### Installation

Clone this repository and install the dependencies. The script itself contains the necessary `pip` commands at the top.

```bash
git clone https://github.com/123RohitVarshit/FINETUNED_DEEPSEEK-R1.git
cd FINETUNED_DEEPSEEK-R1
pip install -r requirements.txt 
```

### Configuration

The script is designed to work seamlessly in Google Colab, where you can store your API tokens as "Secrets" named `Hugging_Face_Token` and `wnb`. If running locally, you might need to adjust the code to load your tokens from environment variables or another secure method.

### Run the Experiment

Simply execute the Python script:

```bash
python finetune_deepseek.py
```

Watch as the model trains and then compare its "before" and "after" answers to see the power of fine-tuning!

## Make It Your Own 🎨

This project is a great starting point. Feel free to experiment:

*   **Use a different model**: Swap `unsloth/DeepSeek-R1-Distill-Llama-8B` with another Unsloth-supported model.
*   **Train on more data**: Increase the number of samples from the dataset by changing `split="train[0:500]"`.
*   **Tweak the training**: Adjust hyperparameters like `learning_rate` or `max_steps` in the `TrainingArguments` to see how it affects the final result.
*   **Change the prompt**: Experiment with different `prompt_style` formats to guide the model in new ways.
