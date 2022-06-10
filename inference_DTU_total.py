from visual_caption import caption_onnx
from machine_translation.run import Inference
import argparse

PREFIX = {
    "de": 'translate English to German: ',
    "fr": 'translate English to French: ',
    "ro": 'translate English to Romanian: ',
}

parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')

# parser.add_argument('--img', default='Flicker8k_Dataset/667626_18933d713e.jpg', help='path to image')
parser.add_argument('--img', default='test_internet_img.jpg', help='path to image')
parser.add_argument('--model',
                    default='visual_caption/1BEST_checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar',
                    help='path to model')
parser.add_argument('--word_map', default='visual_caption/WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json',
                    help='path to word map JSON')
parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')
parser.add_argument('--language', default="zh", choices=["zh", "de", "fr", "ro", "en"])
parser.add_argument('--type', default="DTU", choices=["torch", "DTU", "onnx"])
args = parser.parse_args()

# Generate caption
caption = caption_onnx.run(args)

if args.language == "en":
    print(f"Caption in {args.language}: {caption}")
else:
    # machine translation
    if args.language == "zh":
        pretrain_model = "Helsinki-NLP/opus-mt-en-zh"
    else:
        pretrain_model = "t5-base"
        caption = PREFIX[args.language] + caption

    inference = Inference(args.type, pretrain_model)
    translate_caption = inference.inference(caption)
    print(f"Caption in {args.language}: {translate_caption}")
