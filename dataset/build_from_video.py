"""
从视频中截取画面来生成数据集
 - 场景检测
 - 截图，并做该视频范围内的【重复检测】
 - 模糊，低画质，无效画面去除；（尽量做自动化一点）

"""
from scenedetect import detect, ContentDetector, split_video_ffmpeg, AdaptiveDetector, ThresholdDetector
import os, argparse, json, random
import datetime, time, logging, traceback
from shutil import move
# import subprocess

import cv2
import numpy as np

# scene_list = detect('my_video.mp4', ContentDetector())
# split_video_ffmpeg('my_video.mp4', scene_list)

def timestr():
  return datetime.datetime.now().strftime("%Y%m%d_%H%M-%S") + f"_{random.randint(0, 1000)}" 

imgMap = {}
muMap = {}
sigmaMap = {}
kernel = cv2.getGaussianKernel(11, 1.5)
window = np.outer(kernel, kernel.transpose())

def clearCach():
  imgMap.clear()
  muMap.clear()
  sigmaMap.clear()

# implimentation from scratch
def compute_ssim(key1, key2, source1, source2, resize=True, destWidth=400):
    # print("calc", key1, key2)
    img1 = imgMap.get(key1, cv2.cvtColor(source1, cv2.COLOR_BGR2GRAY)) 
    img2 = imgMap.get(key2, cv2.cvtColor(source2, cv2.COLOR_BGR2GRAY))

    if resize:
      height, width = img1.shape
      newheight = int(height/width * destWidth)
      img1 = cv2.resize(img1, (destWidth, newheight))
      img2 = cv2.resize(img2, (destWidth, newheight))

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    # make sure they have same dimention
    # if img1.shape[1] != img2.shape[1] or img1.shape[0] != img2.shape[0]:
    #     return 0

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    mu1 = muMap.get(key1, cv2.filter2D(img1, -1, window)[5:-5, 5:-5])
    mu2 = muMap.get(key2, cv2.filter2D(img2, -1, window)[5:-5, 5:-5]) 
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = sigmaMap.get(key1, cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq)
    sigma2_sq = sigmaMap.get(key2, cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq)
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    # save to cache
    imgMap[key1] = img1
    imgMap[key2] = img2
    muMap[key1] = mu1
    muMap[key2] = mu2
    sigmaMap[key1] = sigma1_sq
    sigmaMap[key2] = sigma2_sq

    # # debug only
    # save1 = os.path.join("/Users/teli/www/ml/_honor_test/neural_photos/dataset", f"{key1}.jpg")
    # save2 = os.path.join("/Users/teli/www/ml/_honor_test/neural_photos/dataset", f"{key2}.jpg")
    # if not os.path.exists(save1):
    #   cv2.imwrite(save1, img1)
    # if not os.path.exists(save2):
    #   cv2.imwrite(save2, img2)

    return ssim_map.mean()

def getDuration(file):
  data = cv2.VideoCapture(file) 
  # count the number of frames 
  frames = data.get(cv2.CAP_PROP_FRAME_COUNT) 
  fps = data.get(cv2.CAP_PROP_FPS) 
  data.release()
  return frames / fps

# 如果黑条占比太大，比如横屏视频里来了一个竖屏片段
def calculate_black_percentage(gray):
  # Convert the image to grayscale
  # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  black_pixels = np.count_nonzero(gray == 0)

  # Calculate the percentage of black pixels
  total_pixels = gray.shape[0] * gray.shape[1]
  black_percentage = (black_pixels / total_pixels)

  return black_percentage

def removeSimilarFrame(frames, thresh, blurThresh=120):
  count = len(frames)
  ijmap = {}
  removedIdxs = []
  prefixkey = random.randint(0, 1000)
  for ii in range(count):
    if ii in removedIdxs:
      continue
    source = frames[ii]
    for jj in range(count):
      if jj in removedIdxs or ii == jj:
        continue
      # 只考虑前后10个样本，减少计算量
      if abs(ii - jj) > 20:
        continue
      dest = frames[jj]
      cached = ijmap.get(f"{ii}_{jj}", None)
      key1 = f"{prefixkey}_{ii}"
      key2 = f"{prefixkey}_{jj}"
      ssim = cached or compute_ssim(key1, key2, source, dest)
      # print("ssm", ssim, key1, key2)
      if ssim >= thresh:
        removedIdxs.append(jj)
      if cached is None:
        ijmap[f"{jj}_{ii}"] = ssim

  # ---------- 画面质量检测 -----
  for ii, frame in enumerate(frames):
    if ii in removedIdxs:
      continue
    greyImg = imgMap.get(f"{prefixkey}_{ii}", cv2.cvtColor(frames[ii], cv2.COLOR_BGR2GRAY))
    score = cv2.Laplacian(frame, cv2.CV_64F).var()
    # print(score)
    if score < blurThresh:
      print("------ blurry image detected", score, ii)
      removedIdxs.append(ii)
      continue
    # dark images or all white images
    meanpercent = np.mean(greyImg) / 255
    if meanpercent > 0.9 or meanpercent < 0.1:
      print("***** dark or bright image detected", meanpercent, ii)
      removedIdxs.append(ii)
      continue
    # -------- 黑条占比太多 ----------
    blackPercent = calculate_black_percentage(greyImg)
    if blackPercent > 0.4:
      print("~~~~ big black bar", blackPercent, ii)
      removedIdxs.append(ii)
      continue

  # keep the unique frames
  clearCach()
  return removedIdxs

def delete_folder_content(folder_path):
  for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    try:
      if os.path.isfile(file_path) or os.path.islink(file_path):
        os.unlink(file_path)
      elif os.path.isdir(file_path):
        shutil.rmtree(file_path)
    except Exception as e:
      print('Failed to delete %s. Reason: %s' % (file_path, e))

def find_mp4_files(directory):
  for root, dirs, files in os.walk(directory):
    for file in files:
      # only video not in segments
      if file.endswith('.mp4') and root.find("segments") == -1 and file[0] != ".":
        yield os.path.join(root, file)

def getShotTimesFromRange(start, end, snapGap = 2, timePrefix = 0.5):
  duration = end - start - timePrefix
  if duration <= 0:
      return []
      
  count = max(1, duration // snapGap)
  if count == 1:
      return [ start + duration / 2 ]    
  items = np.linspace(start + timePrefix, end, num = int(count), endpoint = False)
  return items.tolist()

def getFrames(video, points):
  frames = []
  idx = 0
  cap = cv2.VideoCapture(video)
  if (cap.isOpened()== False): 
    print("Error opening video stream or file", video)
    return frames

  # Read until video is completed
  while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
      ms = cap.get(cv2.CAP_PROP_POS_MSEC)
      currentCutTime = points[idx] * 1000
      # Check if the current frame is the one we want to save
      if ms >= currentCutTime:
        frames.append(frame)
        idx += 1
        # this is the last point
        if idx > len(points) - 1:
          break
  
    # Break the loop when the video is over
    else: 
      break
  
  # When everything done, release the video capture object
  cap.release()
  # Closes all the frames
  cv2.destroyAllWindows()
  
  return frames

def parse_one_video(file, args, tryCount=0):
  detector = AdaptiveDetector(adaptive_threshold=args.thresha, window_width=4, min_scene_len=15)
  if tryCount > 0:
    detector = ContentDetector(threshold=args.threshc)

  scene_list = detect(file, detector, show_progress=True)
  snappoints = []
  for i, scene in enumerate(scene_list):
    ss = scene[0].get_seconds()
    ee = scene[1].get_seconds()
    snappoints += getShotTimesFromRange(ss, ee, 3)

  if len(snappoints) == 0:
    dd = getDuration(file)
    if not dd or dd < 1:
      return
    snappoints.append(dd / 2)
    # print("!!!no cut point find")
    # if tryCount == 0:
    #   # 换个detector再来一次
    #   return parse_one_video(file, args, tryCount=1)
    # else: # 一个短视频没有裁剪点
    #   return 

  rawframes = getFrames(file, snappoints)
  removedIdxs = removeSimilarFrame(rawframes, thresh=args.simThresh)
  print("removedIdxs", removedIdxs, "rawframes", len(rawframes))
  frames = [ frame for idx, frame in enumerate(rawframes) if idx not in removedIdxs]
  timepoints = [ p for idx, p in enumerate(snappoints) if idx not in removedIdxs]

  basename = file.rsplit("/", 1)[1]
  for ii, frame in enumerate(frames):
    timepoint = timepoints[ii]
    filename = basename.rsplit(".", 1)[0][:20] + f"_{int(timepoint)}_{timestr()}.jpg"
    cv2.imwrite(os.path.join(args.dest, filename), frame)
  print(f"done writing {len(frames)} images!")
  # delete this video
  if not args.keep:
    os.remove(file)

def main():
  parser = argparse.ArgumentParser(description='递归扫描文件夹，找到里面的视频')
  parser.add_argument('dir', metavar='N', type=str, nargs='+',
                      help='folder path that contains mp4 video file')
  parser.add_argument('--threshc', type=int, default=27,
                      help='ContentDetector threshold default 27, Threshold the average change in pixel intensity must exceed to trigger a cut.')
  # parser.add_argument('--thresht', type=int, default=200,
  #                     help='8-bit intensity value that each pixel value (R, G, and B) must be <= to in order to trigger a fade in/out')
  parser.add_argument('--thresha', type=float, default=3.0,
                      help='Threshold (float) that score ratio must exceed to trigger a new scene (see frame metric adaptive_ratio in stats file')
  parser.add_argument('--dest', type=str, default="/Users/teli/www/ml/_honor_test/neural_photos/dataset/snapshots3/",
                      help='where to save the snapshot')
  parser.add_argument('--simThresh', type=float, default=0.3,
                      help='当画面相似度大于多少时，不要这个截图')
  parser.add_argument('--keep', action='store_true', 
                      help='不删除删除处理过的视频文件')
  args = parser.parse_args()

  for directory in args.dir:  
    files = list(find_mp4_files(directory))
    print(" >>> directory:", directory, "video count:", len(files))

    for idx, file in enumerate(files):
      emptyplaceholder = os.path.join(os.path.dirname(file), "done.txt")
      if os.path.exists(emptyplaceholder):
        print("skip")
        continue

      print(f"[{idx}/{len(files)}] {file}")
      try:
        parse_one_video(file, args)
        # pySceneDetect有BUG，很多时候没有检测到任何裁剪点
        time.sleep(0.5)
        # 在这个文件夹里添加一个空文件，证明已经处理过了
        # with open(emptyplaceholder, "w") as ff:
        #   ff.write("")
      except Exception as e:
        logging.error(traceback.format_exc())
        # 如果有问题就移动到同文件夹下 _issue
        issueDir = os.path.join(os.path.dirname(file), "_issue")
        if not os.path.exists(issueDir):
          os.mkdir(issueDir)
        move(file, os.path.join(issueDir, os.path.basename(file)))
      

  print("done!")

if __name__ == "__main__":
  main()