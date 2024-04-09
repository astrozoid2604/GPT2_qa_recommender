import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import time



def generate_text(query, tokenizer, model):
    # Display generated answer
    def print_slowly(text, delay=0.1):
        """Prints out text with a delay between each word to simulate slower typing."""
        for word in text.split():
            print(word, end=' ', flush=True)
            time.sleep(delay)
        print() 
        
    # Encode the query
    input_ids = tokenizer.encode(query, return_tensors='pt')

    # Create the attention_mask as a tensor
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

    # Generate response with adjusted parameters to mitigate repetition and encourage focus
    output = model.generate(
        input_ids,
        max_length=100,                       # Adjusted to a reasonable length to prevent excessive generation
        num_return_sequences=1,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,  # Explicitly set pad_token_id
        temperature=0.7,                      # Slightly lower to reduce randomness
        top_p=0.9,                            # Nucleus sampling for diversity without going off-topic
        repetition_penalty=1.2,               # Penalize repetition to encourage diversity
        do_sample=True,                       # Enable sampling to use temperature and top_p
        early_stopping=True,                  # Stop generation when the model predicts the EOS token
    )

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print_slowly(generated_text, 0.1)
    


# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("/Users/jameslim/Downloads/dataset/GPT2_tokenizer/")
model = GPT2LMHeadModel.from_pretrained("/Users/jameslim/Downloads/dataset/GPT2_finetuned_model_setting2/")

while(True):
    user_query = input("\n\n\nPlease enter a query: ")
    generate_text(user_query, tokenizer, model)
