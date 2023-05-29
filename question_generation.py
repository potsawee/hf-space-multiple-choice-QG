import torch
import re

@torch.no_grad()
def question_generation_sampling(
    g1_model,
    g1_tokenizer,
    g2_model,
    g2_tokenizer,
    context,
    num_questions,
    device,
):
    qa_input_ids = prepare_qa_input(
            g1_tokenizer,
            context=context,
            device=device,
    )
    max_repeated_sampling = int(num_questions * 1.5) # sometimes generated question+answer is invalid
    num_valid_questions = 0
    questions = []
    for q_ in range(max_repeated_sampling):
        # Stage G.1: question+answer generation
        outputs = g1_model.generate(
            qa_input_ids,
            max_new_tokens=128,
            do_sample=True,
        )
        question_answer = g1_tokenizer.decode(outputs[0], skip_special_tokens=False)
        question_answer = question_answer.replace(g1_tokenizer.pad_token, "").replace(g1_tokenizer.eos_token, "")
        question_answer_split = question_answer.split(g1_tokenizer.sep_token)
        if len(question_answer_split) == 2:
            # valid Question + Annswer output
            num_valid_questions += 1
        else:
            continue
        question = question_answer_split[0].strip()
        answer = question_answer_split[1].strip()

        # Stage G.2: Distractor Generation
        distractor_input_ids = prepare_distractor_input(
            g2_tokenizer,
            context = context,
            question = question,
            answer = answer,
            device = device,
            separator = g2_tokenizer.sep_token,
        )
        outputs = g2_model.generate(
            distractor_input_ids,
            max_new_tokens=128,
            do_sample=True,
        )
        distractors = g2_tokenizer.decode(outputs[0], skip_special_tokens=False)
        distractors = distractors.replace(g2_tokenizer.pad_token, "").replace(g2_tokenizer.eos_token, "")
        distractors = re.sub("<extra\S+>", g2_tokenizer.sep_token, distractors)
        distractors = [y.strip() for y in distractors.split(g2_tokenizer.sep_token)]
        options = [answer] + distractors

        while len(options) < 4:
            options.append(options[-1])

        question_item = {
            'question': question,
            'options': options,
        }
        questions.append(question_item)
        if num_valid_questions == num_questions:
            break
    return questions


def prepare_qa_input(t5_tokenizer, context, device):
    """
    input: context
    output: question <sep> answer
    """
    encoding = t5_tokenizer(
        [context],
        return_tensors="pt",
    )
    input_ids = encoding.input_ids.to(device)
    return input_ids


def prepare_distractor_input(t5_tokenizer, context, question, answer, device, separator='<sep>'):
    """
    input: question <sep> answer <sep> article
    output: distractor1 <sep> distractor2 <sep> distractor3
    """
    input_text = question + ' ' + separator + ' ' + answer + ' ' + separator + ' ' + context
    encoding = t5_tokenizer(
        [input_text],
        return_tensors="pt",
    )
    input_ids = encoding.input_ids.to(device)
    return input_ids
