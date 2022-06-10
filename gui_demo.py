import tkinter as tk
from tkinter import filedialog, messagebox, ttk, font
from PIL import Image, ImageTk
from machine_translation.run import Inference
import onnxruntime
import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
from imageio import imread

def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    # img = imresize(img, (256, 256))

    img = np.array(Image.fromarray(img).resize((256, 256)))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    # random_input = torch.randn(32, 3, 256, 256, device='cpu')
    # encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    image_np = image.numpy()
    encoder_output = encoder.run(['output'], input_feed={'input': image_np})
    encoder_out = torch.tensor(encoder_output[0]).to(device)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    # print('print shape: %s', encoder_out.shape)
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds.long()], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds.long()], alpha[prev_word_inds.long()].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds].long()]
        c = c[prev_word_inds[incomplete_inds].long()]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds].long()]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return seq, alphas

def read_image():
    selected_file = filedialog.askopenfilename()
    if len(selected_file) > 0:
        try:
            image = Image.open(selected_file)
        except Exception as e:
            messagebox.showerror('ERROR', e)
            return
        new_width, new_height = image_size[0], image_size[1]
        if image.width != new_width or image.height != new_height:
            factor_temp = min(new_width / image.width, new_height / image.height)
            img_shape = (int(image.width * factor_temp + 0.5), int(image.height * factor_temp + 0.5))
            image = image.resize(img_shape, Image.BILINEAR)
        img_show = ImageTk.PhotoImage(image)
        image_label.config(image=img_show)
        image_label.image=img_show
        image_label.path = selected_file
        answer.set('The answer will be showed here')

def image_caption():
    if not hasattr(image_label, 'path'):
        messagebox.showwarning('WARNING', 'Please select an image before getting the caption')
    else:
        target_language = langauge_var.get()
        seq, _ = caption_image_beam_search(encoder, decoder, image_label.path, word_map, beam_size=5)
        words = [rev_word_map[ind] for ind in seq]
        caption_results = ''
        for word in words:
            if word == '<start>':
                continue
            if word == '<end>':
                continue
            caption_results += word
            caption_results += ' '
        caption_results = caption_results.rstrip().capitalize() + '.'
        if target_language == 'English':
            translation_results = caption_results
        elif target_language == 'Chinese':
            translation_results = zn_inference.inference(caption_results)[0]
            translation_results = translation_results.replace(' ', '')
        else:
            caption_results = f'translate English to {target_language}' + caption_results
            translation_results = t5_inference.inference(caption_results)[0]
        answer.set(translation_results)

def clear():
    answer.set('The answer will be showed here')
    image = Image.new('RGB', image_size, (192, 192, 192))
    image = ImageTk.PhotoImage(image)
    image_label.config(image=image)
    image_label.image = image
    if hasattr(image_label, 'path'):
        del image_label.path

window_size = [960, 720]
image_size = [500, 375]
selected_font = ('Times New Roman', 10, 'bold')
model_path = 'visual_caption/1BEST_checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar'
encoder_onnx_path = 'onnx/encoder.onnx'
word_map_path = 'visual_caption/WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json'
device = torch.device('cpu')

encoder = onnxruntime.InferenceSession(encoder_onnx_path)
decoder = torch.load(model_path)['decoder'].to(device)
with open(word_map_path, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}

zn_inference = Inference('torch', 'Helsinki-NLP/opus-mt-en-zh')
t5_inference = Inference('torch', 't5-base')

root = tk.Tk()
root.config(background='white')
root.title('Image Captioning Demo')
root.geometry(f'{window_size[0]}x{window_size[1]}')
root.minsize(window_size[0], window_size[1])
root.maxsize(window_size[0], window_size[1])
font.families()

image_btn = tk.Button(root, text='Select Image', width=15, command=read_image, font=selected_font, bg='lightblue')
image_btn.place(x=650, y=200)
caption_btn = tk.Button(root, text='Get Caption', width=15, command=image_caption, font=selected_font, bg='lightblue')
caption_btn.place(x=650, y=300)
clear_btn = tk.Button(root, text='Clear', width=15, command=clear, font=selected_font, bg='lightblue')
clear_btn.place(x=650, y=400)

show_label1 = tk.Label(root, text='Image', width=45, font=selected_font, relief=tk.RIDGE)
show_label1.place(x=20, y=50)
show_label2 = tk.Label(root, text='Caption', width=45, font=selected_font, relief=tk.RIDGE)
show_label2.place(x=20, y=500)

image_label = tk.Label(root)
image_label.place(x=20, y=75)
image = Image.new('RGB', image_size, (192, 192, 192))
image = ImageTk.PhotoImage(image)
image_label.config(image=image)
image_label.image = image

langauge_var = tk.StringVar()
langauge_btn = ttk.Combobox(root, width=17, textvariable=langauge_var, state='readonly', font=selected_font)
langauge_btn['values'] = (
    'Chinese',
    'English',
    'French',
    'German',
    'Romanian'
)
langauge_btn.place(x=650, y=100)
langauge_btn.current(0)
answer = tk.StringVar(value='The answer will be showed here')
answer_label = tk.Label(root, textvariable=answer, bg='lightgray', width=45, height=5, wraplength=500, font=selected_font)
answer_label.place(x=20, y=525)

root.mainloop()