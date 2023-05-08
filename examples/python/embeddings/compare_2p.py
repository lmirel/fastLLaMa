
from fastllama import Model
import sys
import glob
import math

# MODEL_PATH = "./ggml-model13b-q4_0.bin"
# MODEL_PATH = "./ggml-model7b-q4_0_nat_enh.bin"
MODEL_PATH = "./ggml-alpaca-7b-q4.bin"

if len(sys.argv) < 3:
  print("run with: " + sys.argv[0] + ' "<phrase1>" "<phrase2>"')
  exit(1)
def stream_token(x: str) -> None:
    """
    This function is called by the llama library to stream tokens
    """
    print(x, end='', flush=True)

model = Model(
        path=MODEL_PATH, #path to model
        num_threads=8, #number of threads to use
        n_ctx=512, #context size of model
        last_n_size=64, #size of last n tokens (used for repetition penalty) (Optional)
        seed=0, #seed for random number generator (Optional)
        embedding_eval_enabled=True,
    )

## phrase 2
print('\nIngesting model with phrase 1: "' + sys.argv[1] + '"')
res = model.ingest(sys.argv[1], is_system_prompt=False) #ingest model with prompt

if res != True:
    print("\nFailed to ingest 1st phrase")
    exit(1)

print("\nGenerating output and embeddings...")
print("")

res = model.generate(
    num_tokens=100, 
    top_p=0.95, #top p sampling (Optional)
    temp=0.8, #temperature (Optional)
    repeat_penalty=1.0, #repetition penalty (Optional)
    streaming_fn=stream_token, #streaming function
    # stop_words=[""] #stop generation when this word is encountered (Optional)
    )
if res != True:
    print("\nFailed to generate response")
    exit(1)

embds1 = model.get_embeddings()
#print ("Embeddings: " + str(embds))

## phrase 2
print('\n\nIngesting model with phrase 2: "' + sys.argv[2] + '"')
res = model.ingest(sys.argv[2], is_system_prompt=False) #ingest model with prompt

if res != True:
    print("\nFailed to ingest 2nd phrase")
    exit(1)

print("\nGenerating output and embeddings...")
print("")

res = model.generate(
    num_tokens=100, 
    top_p=0.95, #top p sampling (Optional)
    temp=0.8, #temperature (Optional)
    repeat_penalty=1.0, #repetition penalty (Optional)
    streaming_fn=stream_token, #streaming function
    # stop_words=[""] #stop generation when this word is encountered (Optional)
    )
if res != True:
    print("\nFailed to generate response")
    exit(1)

embds2 = model.get_embeddings()
#print ("Embeddings: " + str(embds))
print("\n\nDistance between embeddings for the 2 phrases is:" + str(math.dist(embds1, embds2)))
