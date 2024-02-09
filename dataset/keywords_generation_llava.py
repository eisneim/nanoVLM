from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
import base64, time, os, argparse
import random, string

chat_handler = Llava15ChatHandler(clip_model_path="/Users/teli/www/ml/vlm/llava_use/mmproj-model-f16.gguf")
llm = Llama(
  # model_path="/Users/teli/www/ml/vlm/llava_use/llava-13b-f16.gguf",
  model_path="/Users/teli/www/ml/vlm/llava_use/llava-13b-q5_k.gguf",
  chat_handler=chat_handler,
  n_ctx=4096, # n_ctx should be increased to accomodate the image embedding
  logits_all=True,# needed to make llava work
  n_gpu_layers=-1, # 41
  temperature=0.1
)
# prompt = "what is in the image? is there anything special? please respond with keywords only, seperated by comma, for example: 'happy', 'sunny', 'friendship', 'chill', 'love', 'affection', 'travel' etc. key words should try to cover: main subjects, theme, weather, location, relationships, events, activities, behavior, mood, emotions"
prompt = "please describe this image with keywords seperated by comma, keywords should try to cover: main subjects, theme, weather, location, relationships, events, activities, behavior, mood, emotions; please only respond with keywords seperated by ',' and each one should be one or two words"


def random_str(count):
    characters = string.ascii_letters + string.digits + '-_~'
    return ''.join(random.choice(characters) for i in range(count))

def find_image_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if (file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png')) and file[0] != ".":
                yield os.path.join(root, file)

def image_to_base64_data_uri(file_path):
  with open(file_path, "rb") as img_file:
    base64_data = base64.b64encode(img_file.read()).decode('utf-8')
    return f"data:image/png;base64,{base64_data}"

def parse_one_file(file, args):
  startTime = time.time()
  res = llm.create_chat_completion(
    messages = [
      {"role": "system", "content": "You are an assistant who perfectly describes images."},
      {
        "role": "user",
        "content": [
            {"type": "image_url", 
                "image_url": {"url": image_to_base64_data_uri(file)}
            },
            {"type" : "text", "text": prompt }
        ]
      }
    ]
  )
  text = res["choices"][0]["message"]["content"].strip().replace(".", "")
  short = text.replace(", ", ",").replace(" ", "_").lower()
  randNum = 2
  # max allowd filename is 255 on mac
  newname = random_str(randNum) + "=" + short[0 : 255 - 5 - randNum] + ".jpg"
  dest = os.path.join(os.path.dirname(file), newname)
  # prevent same name, just in case 🙂
  if os.path.exists(dest):
    newname = newname.replace(newname[0:randNum], random_str(randNum))
    dest = os.path.join(os.path.dirname(file), newname)

  os.rename(file, dest)

  print(newname)
  print(f"cost: {time.time() - startTime :.2f}s", res["usage"])


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='扫描文件夹得到里面的所有图片并利用VLM用关键字重命名图片文件')
  parser.add_argument('dir', metavar='N', type=str, nargs='+',
                      help='image dirs')
  # parser.add_argument('--batch_size', type=int, default=64,
  #                     help='batch_size')
  # parser.add_argument('--iteration', type=int, default=1000,
  #                     help='iteration')
  # parser.add_argument('--save_frequency', type=int, default=100,
  #                     help='save_frequency')
  # parser.add_argument('--model', type=str, default="resnet50",
  #                     help='选择什么模型:torchvision支持的很多模型')

  args = parser.parse_args()

  for directory in args.dir:
    files = list(find_image_files(directory))
    # random.shuffle(files)

    for ii, file in enumerate(files):
      # name, extension = file.rsplit('.', 1)
      if os.path.basename(file)[2] == "=":
        print("skip", file)
        continue

      print(f"[{ii} / {len(files)}] {file}")
      try:
        parse_one_file(file, args)
      except Exception as e:
        print("err file:", file)

  print("done")
