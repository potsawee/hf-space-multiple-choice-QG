import gradio as gr
import random
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from question_generation import question_generation_sampling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

g1_tokenizer = AutoTokenizer.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer")
g1_model = AutoModelForSeq2SeqLM.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer")
g2_tokenizer = AutoTokenizer.from_pretrained("potsawee/t5-large-generation-race-Distractor")
g2_model = AutoModelForSeq2SeqLM.from_pretrained("potsawee/t5-large-generation-race-Distractor")
g1_model.eval()
g2_model.eval()
g1_model.to(device)
g2_model.to(device)


def generate_multiple_choice_question(
    context
):
    num_questions = 1
    question_item = question_generation_sampling(
        g1_model, g1_tokenizer,
        g2_model, g2_tokenizer,
        context, num_questions, device
    )[0]
    question = question_item['question']
    options = question_item['options']
    options[0] = f"{options[0]} [ANSWER]"
    random.shuffle(options)
    output_string = f"Question: {question}\n[A] {options[0]}\n[B] {options[1]}\n[C] {options[2]}\n[D] {options[3]}"
    return output_string

demo = gr.Interface(
    fn=generate_multiple_choice_question,
    inputs=gr.Textbox(lines=5, placeholder="Context Here..."),
    outputs=gr.Textbox(lines=5, placeholder="Question: ...\n[A] ...\n[B] ...\n[C] ...\n[D] ..."),
)
demo.launch()
