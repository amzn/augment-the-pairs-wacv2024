"""
  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
  
  Licensed under the Apache License, Version 2.0 (the "License").
  You may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  
      http://www.apache.org/licenses/LICENSE-2.0
  
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""

import torchvision.transforms.functional as F

left_right_dict = {
    "left": {"left", "leftmost", "bottomleft", "leftside", "farleft", "leftest", "leftiest", "upleft", "leftier", "upperleft", "topleft", "lefty", "leftmiddle"},
    "right": {"right", "rightmost", "bottomright", "rightside", "farright", "rightest", "rightiest", "upright", "rightier", "upperright", "topright", "righty", "rightmiddle"}
}

def get_phrases(positive_tokens, caption):
    phrases = []
    for token_list in positive_tokens:
        if not any(token_list):
            continue
        for (l, r) in token_list:
            phrases.append(caption[l:r])
    return phrases
        
def is_letter(letter):
    return letter>='a' and letter<='z'

def find_words(phrase):
    phrase = phrase.lower() # lowercase only 
    word_idxes = []
    i = 0
    left = i
    while i <len(phrase):
        if is_letter(phrase[i]):
            i += 1
            continue
        if left != i:
            word_idxes.append((left, i))
        i += 1
        left = i
        
    if left != i:
        word_idxes.append((left, i))
    
    words = [phrase[l:r] for l,r in word_idxes]
    return words


def get_unique_tokens(tokens_positive):
    unique_tokens = set()
    out_tokens = []
    for token in tokens_positive:
        if str(token) not in unique_tokens:
            out_tokens.append(token)
            unique_tokens.add(str(token))
    return sorted(out_tokens)

def find_and_replace_left_right(subcaption):
    # [assumption]: 
    # (1) one word (left or right) in phrase 
    # (2) left and right can not appear in the same phrase
    if 'left' in subcaption:
        words = find_words(subcaption) 
        for keyword in left_right_dict['left']:
            if keyword in words:
                keyword_neg = keyword.replace('left', 'right')
                subcaption = subcaption.replace(keyword, keyword_neg)
                break
    elif 'right' in subcaption:
        words = find_words(subcaption) 
        for keyword in left_right_dict['right']:
            if keyword in words:
                keyword_neg = keyword.replace('right', 'left')
                subcaption = subcaption.replace(keyword, keyword_neg)
                break
    return subcaption
    
        
def tokens_hflip(unique_tokens, caption):
    """
    Flip position word in caption, and correct corresponding bbox-related token positions 
    """
    new_caption = ''
    last_idx = 0
    oldToken2NewToken = {}
    for token_list in unique_tokens:
        if not any(token_list):
            continue
        new_sub_tokens = []
        for (beg_, end_) in token_list:
            # step1: flip gap caption between last_idx and tokens, for example, woman on the right; right is not in token
            last_gap_caption = caption[last_idx:beg_]
            last_idx += len(last_gap_caption)
            if len(last_gap_caption) >= 4:
                last_gap_caption = find_and_replace_left_right(last_gap_caption)
            new_caption += last_gap_caption
            
            # step2: subcaption within token 
            subcaption = caption[beg_:end_]
            last_idx += len(subcaption)
            if len(subcaption) >= 4:
                subcaption = find_and_replace_left_right(subcaption)
            left = len(new_caption)
            new_caption += subcaption
            right = len(new_caption)
            new_sub_tokens.append([left, right])
        oldToken2NewToken[str(token_list)] = new_sub_tokens # create mapping between new and old tokens
        
    if last_idx < len(caption)-1:
        subcaption = caption[last_idx:]
        if len(subcaption) >= 4:
            subcaption = find_and_replace_left_right(subcaption)
        new_caption += subcaption
    return new_caption, oldToken2NewToken
            
def has_left_right(caption):
    return 'left' in caption or 'right' in caption

                
def has_left_right_in_dict(caption):
    words = find_words(caption)
    for word in words:
        if 'left' in word:
            if word in left_right_dict['left']:
                return True
        if 'right' in word:
            if word in left_right_dict['right']:
                return True
    return False
    
def thflip(image, target):
    """
    Text Conditioned Horizontal Flip (Rule Based)
    Input:
        image (np.array): H x W x C 
        target (dict): {'caption': str caption,
                        'boxes': list of boxes with format [x1, y1, x2, y2],
                        'tokens_positive': list of phrase position indexes in the caption, e.g., [[1,10]],
                        ...}
    Return:
        flipped_image:  (np.array): H x W x C
        target (dict): {'caption': str caption,
                        'boxes': list of boxes with format [x1, y1, x2, y2],
                        'tokens_positive': list of phrase position indexes in the caption, e.g., [[1,10]],
                        ...}
    """
    # Rule1: if caption does not contain 'left' or 'right', 
    # directly apply horizontal flip to image without modifying caption
    if not has_left_right(target['caption']):
        # flip image
        flipped_image = F.hflip(image)
        width, height = image.size
        # flip bbox
        target["boxes"][:, [0,2]] = width-target["boxes"][:, [2,0]]
        return flipped_image, target
    
    # Rule2: skip flip if caption contains 'right' or 'left', 
    # but the corresponding word is not in predefined dictionary
    if not has_left_right_in_dict(target['caption']):
        return image, target

    # Rule3: apply horizontal flip to both image and target
    # flip image
    flipped_image = F.hflip(image)
    width, height = image.size
    
    # flip bbox
    target["boxes"][:, [0,2]] = width-target["boxes"][:, [2,0]]
    
    # correct word tokens
    unique_tokens = get_unique_tokens(target['tokens_positive'])
    
    new_caption, oldToken2NewToken = tokens_hflip(unique_tokens, target['caption'])
    new_tokens_positive = []
    for token in target['tokens_positive']:
        if not any(token):
            new_tokens_positive.append(token)
        else:
            new_tokens_positive.append(oldToken2NewToken[str(token)])
    target['tokens_positive'] = new_tokens_positive
    target['caption'] = new_caption
    
    return flipped_image, target
