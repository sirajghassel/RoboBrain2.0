from typing import Union
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class SimpleInference:
    """
    A class for performing inference using Hugging Face models.
    """
    
    def __init__(self, model_id="BAAI/RoboBrain2.0-7B"):
        """
        Initialize the model and processor.
        
        Args:
            model_id (str): Path or Hugging Face model identifier (default: "BAAI/RoboBrain2.0-7B")
        """
        print("Loading Checkpoint ...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype="auto", 
            device_map="auto"
        )

        self.processor = AutoProcessor.from_pretrained(model_id)
        
    def inference(self, text:str, image: Union[list,str], enable_thinking=False, do_sample=True, temperature=0.7):
        """Perform inference with text and images input."""
        if isinstance(image, str):
            image = [image]
        
        messages = [
            {
                "role": "user",
                "content": [
                    *[
                        {"type": "image", 
                         "image": path if path.startswith("http") else f"file://{path}"
                        } for path in image
                    ],
                    {"type": "text", "text": f"{text}"},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if enable_thinking:
            print("Thinking enabled.")
            text = f"{text}<think>"
        else:
            print("Thinking disabled.")
            text = f"{text}<think></think><answer>"


        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference
        print("Running inference ...")
        generated_ids = self.model.generate(**inputs, max_new_tokens=768, do_sample=do_sample, temperature=temperature)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        if enable_thinking:
            thinking_text = output_text[0].split("</think>")[0].replace("<think>", "").strip()
            answer_text = output_text[0].split("</think>")[1].replace("<answer>", "").replace("</answer>", "").strip()
        else:
            thinking_text = ""
            answer_text = output_text[0].replace("<answer>", "").replace("</answer>", "").strip()

        return {
            "thinking": thinking_text,
            "answer": answer_text
        }


if __name__ == "__main__":

    model = SimpleInference("BAAI/RoboBrain2.0-7B")

    prompt = "What is shown in this image?"
    image = "http://images.cocodataset.org/val2017/000000039769.jpg"

    pred = model.inference(prompt, image, enable_thinking=True, do_sample=True)
    print(f"Prediction:\n{pred}")
