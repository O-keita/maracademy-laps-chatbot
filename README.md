# MarAcademy Computer Science Assistant (Chatbot)

---

## Table of Contents
- [Introduction](#introduction)
- [Dataset Collection & Preprocessing](#dataset-collection--preprocessing)
- [Model Selection & Fine-Tuning](#model-selection--fine-tuning)
- [Performance Evaluation](#performance-evaluation)
- [UI Integration & Functionality](#ui-integration--functionality)
- [Conclusion](#conclusion)
- [Future Improvements](#future-improvements)
- [Quickstart](#quickstart)
- [Project Links](#project-links)
- [References](#references)

---

## Introduction

In the era of digital transformation, personalized learning and actionable analytics are revolutionizing education, making individualized support and guidance more accessible than ever. MarAcademy, with its mission to bridge the gap between ambition and opportunity, is committed to empowering learners through expertly designed courses, hands-on mentorship, and fully funded scholarships in the tech domain.

This project presents the development of a domain-specific chatbot for MarAcademy, designed to assist students with computer science queries, program information, learning analytics, and personalized recommendations. Leveraging the power of Transformer-based models, specifically a fine-tuned GPT-2, the chatbot delivers context-aware, natural language responses tailored to the unique needs of MarAcademy’s student community.

By automating responses to both academic and institutional questions, the MarAcademy chatbot enhances learner engagement, provides timely support, and contributes to a better educational experience.

---

## Dataset Collection & Preprocessing

### Data Sources
To ensure the MarAcademy chatbot delivers accurate, comprehensive, and context-aware answers to students, the project leveraged multiple high-quality, domain-specific datasets:
- **Computer Science Theory QA Dataset**: Covers hardware/software concepts, data structures, algorithms, and more (sourced from Kaggle).
- **MarAcademy Resources**: Custom datasets (`maracademy.json`, `computer_science.json`, `data_science.json`) include curated information about MarAcademy’s mission, vision, courses, scholarships, and mentorship.

### Preprocessing Steps
- **Duplicate Removal:** Ensures data diversity.
- **Handling Missing Values:** Short or incomplete Q&A pairs were removed.
- **Text Normalization:** Converted text to lowercase, trimmed whitespace.
- **Tokenization:** Used GPT-2 Byte-Pair Encoding (BPE) tokenizer.

The cleaned data was saved as `conversations_clean.csv` for downstream model training.

---

## Model Selection & Fine-Tuning

### Model Selection
- **Architecture:** GPT-2 Transformer model.
- **Advantages:** Robust generative capabilities, context-aware responses for both technical and institutional queries.

### Fine-Tuning
- Fine-tuned on a combined dataset of theoretical computer science Q&A and MarAcademy-specific resources.
- **Hyperparameters Tuned:**
  - Epochs: Increased gradually to enhance generalization while avoiding overfitting.
  - Learning Rate: Adjusted to balance stability and speed.
  - Batch Size: Configured based on GPU availability.

### Results
- **Best Model:** GPT-2 (20 epochs, final checkpoint).
- **Deployment:** DistilGPT2 selected due to lower memory usage and real-time inference capabilities.

---

## Performance Evaluation

### Quantitative Metrics
| Model Version       | Epochs | Training Loss | Perplexity |
|---------------------|:------:|:-------------:|:----------:|
| GPT-2 (baseline)    |   3    |     1.80      |    6.05    |
| GPT-2 (mid)         |  10    |     0.40      |    1.49    |
| GPT-2 (final)       |  20    |    0.187      |    1.21    |
| DistilGPT2 (final)  |  20    |    0.785      |    2.19    |

### Qualitative Testing
- **Example Queries:**
  - Q: "Explain the difference between a stack and a queue."
    - A: "A stack follows Last-In-First-Out (LIFO), while a queue follows First-In-First-Out (FIFO)."
  - Q: "What scholarships does MarAcademy offer?"
    - A: "MarAcademy provides fully-funded scholarships to ensure education is accessible to everyone, in partnership with initiatives such as DataCamp Donates."

---

## UI Integration & Functionality

### Deployment
- **Streamlit App:** [https://maracademy-laps-chatbot.streamlit.app/](https://maracademy-laps-chatbot.streamlit.app/)
- **Custom Flask/Tailwind Server:** [https://api.ngaagenticflow.agency/omar/own/chatbot/](https://api.ngaagenticflow.agency/omar/own/chatbot/)

Both UIs access the centralized fine-tuned GPT-2 model hosted on Hugging Face Hub.

### Features
- Real-time Q&A.
- Seamless integration of both institutional and technical queries.
- Out-of-domain handling via prompt engineering.

---

## Conclusion

The MarAcademy chatbot exemplifies a robust, flexible approach to deploying transformer-based conversational AI by leveraging a fine-tuned GPT-2 model. Both the Streamlit and Flask/Tailwind interfaces guarantee a consistent, high-quality user experience, enabling MarAcademy to provide accessible, personalized support to its learners.

---

## Future Improvements
1. **Advanced Out-of-Domain Handling:** Incorporate intent classification models to handle out-of-domain queries more effectively.
2. **Scalability:** Migrate to autoscaling cloud infrastructure for higher concurrent user loads.
3. **Multi-turn Contextual Memory:** Enable the chatbot to remember context across multiple exchanges.
4. **Model Optimization:** Apply quantization for faster inference and reduced memory usage.
5. **Continuous Learning:** Adapt to evolving user needs via periodic re-training.

---

## Quickstart

```bash
# Clone the repository
git clone https://github.com/O-keita/maracademy-laps-chatbot.git
cd maracademy-laps-chatbot

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\Activate.ps1  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app locally
streamlit run ui/app.py
```

---

## Project Links

- **GitHub Repo:** [https://github.com/O-keita/maracademy-laps-chatbot](https://github.com/O-keita/maracademy-laps-chatbot)
- **Hugging Face Model:** [https://huggingface.co/Omar-keita/gpt2-finetuned-maracademy](https://huggingface.co/Omar-keita/gpt2-finetuned-maracademy)
- **Video Demo:** [https://youtu.be/8wvi2IuvipQ](https://youtu.be/8wvi2IuvipQ)

---

## References

1. OpenAI Community. (n.d.). GPT-2. Hugging Face. Retrieved October 16, 2025, from [https://huggingface.co/openai-community/gpt2](https://huggingface.co/openai-community/gpt2)
2. Hugging Face. (n.d.). Hugging Face Hub documentation. Retrieved October 16, 2025, from [https://huggingface.co/docs/hub/en/index](https://huggingface.co/docs/hub/en/index)
3. MarAcademy. (n.d.). MarAcademy – Empowering learners through technology education. Retrieved October 16, 2025, from [https://maracademy.org](https://maracademy.org)