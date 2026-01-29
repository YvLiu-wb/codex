import os
import json
import re
from datetime import datetime
from collections import Counter
# from qwenvl_class import QwenvlModel
from Qwen2_5Vl import QwenvlModel
from qwen2_5_class import QwenModel


log = "/home/yvwenqiang/git_package/penrose/py/logs_test/qwen_sft_1_28"
K_ANSWERS = 3
K_PERTURB = 3


def create_log_folders(base_path):
    log_types = ["correct_logs", "error_logs", "unmatched_logs"]
    question_types = ["Position", "Geometry Shape", "Geometric Relationship"]

    for log_type in log_types:
        if not os.path.exists(os.path.join(base_path, log_type)):
            os.makedirs(os.path.join(base_path, log_type))

        for q_type in question_types:
            if not os.path.exists(os.path.join(base_path, log_type, q_type)):
                os.makedirs(os.path.join(base_path, log_type, q_type))



def log_question(log_type, question_id, question, answer, description, base_path, question_type):
    log_folder = os.path.join(base_path, log_type, question_type)
    log_file_path = os.path.join(
        log_folder, f"{question_id}.txt" if question_id else "unknown_id.txt")

    with open(log_file_path, "w", encoding="utf-8") as log_file:
        log_file.write(
            f"ID: {question_id}\n" if question_id else "ID: unknown\n")
        log_file.write(f"Question: {question}\n")
        log_file.write(f"Answer: {answer}\n")
        log_file.write(f"Description: {description}\n")



def _normalize_answer_item(item):
    return re.sub(r"\s+", "", item.replace("angle ", "∠").replace("_", "")).upper()



def _parse_answer_list(answer):
    return [item.strip() for item in answer.strip("[]").split(",") if item.strip()]



def _parse_model_answers(description):
    processed_desc = description.replace("angle ", "∠").replace("_", "")
    return [item.strip() for item in re.split(r"[\n,;]+", processed_desc) if item.strip()]


def _extract_option_letter(text):
    if not text:
        return None
    boxed_match = re.findall(r"\{([A-D])\}", text.upper())
    if boxed_match:
        return boxed_match[-1]
    standalone_match = re.findall(r"\b([A-D])\b", text.upper())
    if standalone_match:
        return standalone_match[-1]
    last_match = re.findall(r"[A-D]", text.upper())
    return last_match[-1] if last_match else None


def _normalize_question(text):
    return re.sub(r"\s+", " ", text or "").strip()


def _parse_perturbations(raw_text, limit):
    candidates = [
        _normalize_question(item)
        for item in re.split(r"[\n]+", raw_text)
        if _normalize_question(item)
    ]
    unique = []
    seen = set()
    for item in candidates:
        if item not in seen:
            unique.append(item)
            seen.add(item)
        if len(unique) >= limit:
            break
    return unique


def generate_perturbations(model, question, k):
    prompt = (
        "请对以下问题进行语义等价的改写，保持题意与选项不变，"
        f"生成{k}个不同表述的版本。只输出每个版本一行，不要编号或额外说明。\n"
        f"问题：{question}"
    )
    response = model.generate_response(prompt)
    perturbations = _parse_perturbations(response, k)
    return perturbations


def build_question_message(question, image_path, image_examples):
    message = [
        {
            "role": "system",
            "content": (
                "According to the picture, answer the question. "
                "Note that when referring to angles in the answer, "
                "you must use the symbol ∠. "
                f"Return {K_ANSWERS} distinct answers separated by commas, "
                "and do not add any extra text."
            ),
        },
    ]
    message.extend(image_examples)
    message.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image", "image": f"file://{image_path}"},
            ],
        }
    )
    return message


def choose_voted_answer(answer_candidates, fallback_answer):
    option_candidates = [_extract_option_letter(item) for item in answer_candidates]
    option_candidates = [item for item in option_candidates if item]
    if not option_candidates:
        return _extract_option_letter(fallback_answer)
    counts = Counter(option_candidates)
    most_common = counts.most_common()
    if most_common and most_common[0][1] > 1:
        top_count = most_common[0][1]
        top_answers = [item for item, count in most_common if count == top_count]
        if len(top_answers) == 1:
            return top_answers[0]
        return _extract_option_letter(fallback_answer)
    return _extract_option_letter(fallback_answer)


def is_correct_answer(answer, description):
    correct_answers = {
        _normalize_answer_item(item)
        for item in _parse_answer_list(answer)
    }
    model_answers = {
        _normalize_answer_item(item)
        for item in _parse_model_answers(description)
    }
    return bool(correct_answers & model_answers)



def log_summary(log_base_path, question_type_counts):
    # 确保日志目录存在
    if not os.path.exists(log_base_path):
        os.makedirs(log_base_path)

    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join(log_base_path, f"summary_{timestamp}.log")

    summary = {}
    total_correct = 0
    total_questions = 0

    # 计算每个问题类型的准确率
    summary = {}
    for question_type, counts in question_type_counts.items():
        correct = counts["correct"]
        total = counts["total"]
        accuracy = correct / total if total > 0 else 0.0
        summary[question_type] = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
        }
        total_correct += correct
        total_questions += total

    total_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
    summary["total"] = {
        "correct": total_correct,
        "total": total_questions,
        "accuracy": total_accuracy,
    }
    # 将结果写入日志文件
    with open(log_file_path, "w") as log_file:
        json.dump(summary, log_file, indent=4)

    print(f"Summary logged to {log_file_path}")



def process_json_files(folder_path):
    model = QwenvlModel()
    perturb_model = QwenModel()
    correct_count = 0
    total_count = 0

    # Dictionary to store counts for each question type
    question_type_counts = {
        "Position": {"correct": 0, "total": 0},
        "Geometry Shape": {"correct": 0, "total": 0},
        "Geometric Relationship": {"correct": 0, "total": 0},
    }

    # Specify the path where logs should be stored
    log_base_path = log
    create_log_folders(log_base_path)

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            print(f"开始处理文件: {filename}")

            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

                image_path1 = "/home/liubing/penrose/py/Manual_Annotation/image/3854.png"
                image_path2 = "/home/liubing/penrose/py/Manual_Annotation/image/3861.png"
                image_path3 = "/home/liubing/penrose/py/Manual_Annotation/image/1230.png"
                image_path = data.get("image")
                question = data.get("question")
                answer = data.get("answer")
                question_type = data.get("question_type")
                question_id = data.get("id")  # Assuming there's an id field

                if all([image_path, question, answer, question_type, question_id]):
                    image_examples = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Which tangent line intersects with the circle at point D?",
                                },
                                {"type": "image", "image": f"file://{image_path1}"},
                            ],
                        },
                        {"role": "assistant", "content": "AD\n"},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "What point is the intersection point of the tangent AB with circle ⊙O?",
                                },
                                {"type": "image", "image": f"file://{image_path2}"},
                            ],
                        },
                        {"role": "assistant", "content": "P\n"},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "What is the inscribed angle corresponding to the arc BC?",
                                },
                                {"type": "image", "image": f"file://{image_path3}"},
                            ],
                        },
                        {"role": "assistant", "content": "∠BAC"},
                    ]
                    perturb_questions = generate_perturbations(perturb_model, question, K_PERTURB)
                    question_variants = [question] + [
                        item for item in perturb_questions if item != question
                    ]
                    model_answers = []
                    for variant in question_variants:
                        message = build_question_message(variant, image_path, image_examples)
                        description = model.generate(messages=message)
                        model_answers.append(description)

                    voted_answer = choose_voted_answer(model_answers, model_answers[0])
                    correct_option = _extract_option_letter(answer)
                    is_correct = (
                        voted_answer == correct_option
                        if correct_option
                        else is_correct_answer(answer, ",".join(model_answers))
                    )
                    question_type_counts[question_type]["total"] += 1

                    if is_correct:
                        correct_count += 1
                        question_type_counts[question_type]["correct"] += 1
                        log_question(
                            "correct_logs",
                            question_id,
                            question,
                            answer,
                            json.dumps(
                                {
                                    "original_question": question,
                                    "perturbations": perturb_questions,
                                    "model_answers": model_answers,
                                    "voted_answer": voted_answer,
                                },
                                ensure_ascii=False,
                                indent=2,
                            ),
                            log_base_path,
                            question_type,
                        )
                    else:
                        log_question(
                            "error_logs",
                            question_id,
                            question,
                            answer,
                            json.dumps(
                                {
                                    "original_question": question,
                                    "perturbations": perturb_questions,
                                    "model_answers": model_answers,
                                    "voted_answer": voted_answer,
                                },
                                ensure_ascii=False,
                                indent=2,
                            ),
                            log_base_path,
                            question_type,
                        )

                    total_count += 1

                else:
                    print(
                        f"File: {filename} does not contain 'image', 'question', 'answer', 'question_type', or 'id' keys."
                    )
                    log_question(
                        "unmatched_logs",
                        None,
                        question,
                        answer if answer else "N/A",
                        "N/A",
                        log_base_path,
                        question_type if question_type else "Unknown",
                    )

    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    print(f"Total Files Processed: {total_count}")
    print(f"Correct Answers: {correct_count}")
    print(f"Overall Accuracy: {accuracy:.2f}%")

    # Print accuracy for each question type
    for q_type, counts in question_type_counts.items():
        type_accuracy = (
            (counts["correct"] / counts["total"]) * 100
            if counts["total"] > 0
            else 0
        )
        print(
            f"Question Type: {q_type}, Total: {counts['total']}, Correct: {counts['correct']}, Accuracy: {type_accuracy:.2f}%"
        )
    log_summary(log_base_path, question_type_counts)


# Example usage
folder_path = "/home/yvwenqiang/git_package/penrose/py/problem/manual-1"
process_json_files(folder_path)
