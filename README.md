Fine-tuning Llama-2 on Hugging Face 🦙

---

# **Fine-tune Llama-2 using QLoRA**

This repository demonstrates the fine-tuning process of the **Llama-2-7b-chat-hf** model from Hugging Face using **QLoRA** (Quantized Low-Rank Adaptation) for efficient parameter tuning and reduced GPU memory usage. The model is fine-tuned on the **Guanaco Llama 2 dataset** for enhanced performance in causal language modeling tasks.

---

## **Features**
- **Efficient Training**: Utilizes QLoRA to fine-tune the model in 4-bit precision, drastically reducing GPU VRAM usage.
- **Custom Dataset**: Incorporates the **`mlabonne/guanaco-llama2-1k`** dataset with a Llama 2 chat-specific prompt structure.
- **LoRA Adaptation**: Fine-tunes selected layers with LoRA, keeping the base model weights intact.
- **Output Storage**: Trained model and tokenizer are saved locally and can be pushed to the Hugging Face Hub.

---

## **Requirements**

Install the following Python packages:

```bash
pip install accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7
```

---

## **How to Use**

### **1. Clone the repository**
```bash
git clone 
cd 
```

### **2. Prepare your environment**
Install the required libraries:
```bash
pip install -r requirements.txt
```

### **3. Configure the model and dataset**
Edit the following parameters in the notebook or script as needed:
- **`model_name`**: Base model name (e.g., `"NousResearch/Llama-2-7b-chat-hf"`).
- **`dataset_name`**: Dataset for fine-tuning (e.g., `"mlabonne/guanaco-llama2-1k"`).
- **`output_dir`**: Path to save trained model and checkpoints.

### **4. Run the fine-tuning process**
Execute the fine-tuning process:
```bash
python fine_tune_llama2.py
```

### **5. Monitor training with TensorBoard**
Start TensorBoard to monitor the training process:
```bash
tensorboard --logdir results/runs
```

---

## **Example: Text Generation**

After fine-tuning, test the model with a prompt:
```python
from transformers import pipeline

pipe = pipeline(task="text-generation", model="path/to/fine-tuned-model", tokenizer="path/to/tokenizer")
result = pipe("<s>[INST] What is a large language model? [/INST]")
print(result[0]['generated_text'])
```

---

## **Contributing**

Contributions are welcome! If you encounter issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

---

## **License**

This project is licensed under the [MIT License](LICENSE).

---

## **Acknowledgements**

- **Hugging Face** for their powerful libraries and model hub.
- **Meta AI** for developing Llama-2.
- **`mlabonne/guanaco-llama2-1k`** dataset for the fine-tuning process.

