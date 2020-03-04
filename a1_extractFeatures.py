import numpy as np
import argparse
import json
import re
from tqdm import tqdm

# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}

def extract1(comment, BGL_dict, War_dict):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    # Assume the default value for all of the above features to be zero
    feats = np.zeros(29)

    # TODO: Extract features that rely on capitalization.
    # TODO: Lowercase the text in comment. Be careful not to lowercase the tags. (e.g. "Dog/NN" -> "dog/NN").
    # TODO: Extract features that do not rely on capitalization.
    comment = " " + comment
    num_sentences = len(re.findall(r'\n', comment))
    comment = re.sub(r'\s', ' ', comment)

    # 1. Number of words in uppercase (≥ 3 letters long)
    feats[0] = len(re.compile('[A-Z]{3,}\/\S+').findall(comment))
    # 2. Number of first-person pronouns
    feats[1] = len(re.findall(r'\s' + r'/|\s'.join(FIRST_PERSON_PRONOUNS) + '/', comment, re.I))
    # 3. Number of second-person pronouns
    feats[2] = len(re.findall(r'\s' + r'/|\s'.join(SECOND_PERSON_PRONOUNS) + '/', comment, re.I))
    # 4. Number of third-person pronouns
    feats[3] = len(re.findall(r'\s' + r'/|\s'.join(THIRD_PERSON_PRONOUNS) + '/', comment, re.I))
    # 5. Number of coordinating conjunctions
    feats[4] = len(re.compile('(\w+)\/CC').findall(comment))
    # 6. Number of past-tense verbs
    feats[5] = len(re.compile('(\w+)\/VBD').findall(comment))
    # 7. Number of future-tense verbs
    feats[6] = len(re.compile('((\'ll\/MD\w*|will\/MD\w*|gonna\/\w+)\s+\w+\/VB)|(go\/VB\w*\s+to\/TO\w*\s+\w+\/VB)').findall(comment))
    # 8. Number of commas
    feats[7] = len(re.compile('\S+/,').findall(comment))
    # 9. Number of multi-character punctuation tokens
    feats[8] = len(re.compile('([?!,;:\.\-`"]{2,})\/').findall(comment))
    # 10. Number of common nouns
    feats[9] = len(re.compile('(\w+)\/NNS?\s').findall(comment))
    # 11. Number of proper nouns
    feats[10] = len(re.compile('(\w+)\/NNPS?').findall(comment))
    # 12. Number of adverbs
    feats[11] = len(re.compile('(\w+)\/RB[RS]?').findall(comment))
    # 13. Number of wh- words
    feats[12] = len(re.compile('(\w+)\/W(DT|P|P$|RB)').findall(comment))
    # 14. Number of slang acronyms
    feats[13] = len(re.findall(r'\s' + r'/|\s'.join(SLANG) + '/', comment, re.I))
    # 15. Average length of sentences, in tokens
    if num_sentences != 0:
        feats[14] = len(comment.split())/num_sentences
    # 16. Average length of tokens, excluding punctuation-only tokens, in characters
    search = [x[1:-1].lower() for x in re.findall(r'\s\S*\w\S*/', comment)]
    if len(search) != 0:
        feats[15] = len("".join(search))/len(search)
    # 17. Number of sentences.
    feats[16] = num_sentences
    # BGL norms
    AoA = []
    IMG = []
    FAM = []
    # Warringer norms
    vwar = []
    awar = []
    dwar = []
    search = [x[1:-1].lower() for x in re.findall(r'\s\w+/', comment)]
    for word in search:
        if word in BGL_dict and word != '':
            AoA.append(float(BGL_dict[word]['AoA (100-700)']))
            IMG.append(float(BGL_dict[word]['IMG']))
            FAM.append(float(BGL_dict[word]['FAM']))
        if word in War_dict and word != '':
            vwar.append(float(War_dict[word]['V.Mean.Sum']))
            awar.append(float(War_dict[word]['A.Mean.Sum']))
            dwar.append(float(War_dict[word]['D.Mean.Sum']))
    if len(AoA) == 0:
        feats[17] = 0
        feats[20] = 0
    else:
        # 18. Average of AoA (100-700) from Bristol, Gilhooly, and Logie norms
        feats[17] = np.mean(AoA)
        # 21. Standard deviation of AoA (100-700) from Bristol, Gilhooly, and Logie norms
        feats[20] = np.std(AoA)
    if len(IMG) == 0:
        feats[18] = 0
        feats[21] = 0
    else:
        # 19. Average of IMG from Bristol, Gilhooly, and Logie norms
        feats[18] = np.mean(IMG)
        # 22. Standard deviation of IMG from Bristol, Gilhooly, and Logie norms
        feats[21] = np.std(IMG)
    if len(FAM) == 0:
        feats[19] = 0
        feats[22] = 0
    else:
        # 20. Average of FAM from Bristol, Gilhooly, and Logie norms
        feats[19] = np.mean(FAM)
        # 23. Standard deviation of FAM from Bristol, Gilhooly, and Logie norms
        feats[22] = np.std(FAM)
    if len(vwar) == 0:
        feats[23] = 0
        feats[26] = 0
    else:
        # 24. Average of V.Mean.Sum from Warringer norms
        feats[23] = np.mean(vwar)
        # 27. Standard deviation of V.Mean.Sum from Warringer norms
        feats[26] = np.std(vwar)
    if len(awar) == 0:
        feats[24] = 0
        feats[27] = 0
    else:
        # 25. Average of A.Mean.Sum from Warringer norms
        feats[24] = np.mean(awar)
        # 28. Standard deviation of A.Mean.Sum from Warringer norms
        feats[27] = np.std(awar)
    if len(dwar) == 0:
        feats[25] = 0
        feats[28] = 0
    else:
        # 26. Average of D.Mean.Sum from Warringer norms
        feats[25] = np.mean(dwar)
        # 29. Standard deviation of D.Mean.Sum from Warringer norms
        feats[28] = np.std(dwar)
    return feats

def extract2(feats, comment_class, comment_id, feats_left, feats_center,
feats_right, feats_alt, id_left, id_center, id_right, id_alt):
    ''' This function adds features 30-173 for a single comment.

    Parameters:
        feats: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (this
        function adds feature 30-173). This should be a modified version of
        the parameter feats.
    '''
    if comment_class == "Left":
        feats[29:173] = feats_left[id_left.index(comment_id)]
        feats[173] = 0
    elif comment_class == "Center":
        feats[29:173] = feats_center[id_center.index(comment_id)]
        feats[173] = 1
    elif comment_class == "Right":
        feats[29:173] = feats_right[id_right.index(comment_id)]
        feats[173] = 2
    elif comment_class == "Alt":
        feats[29:173] = feats_alt[id_alt.index(comment_id)]
        feats[173] = 3
    return feats


def main(args):
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173+1))

    # Load the feats and ids
    feats_left = np.load(args.a1_dir + '/feats/Left_feats.dat.npy')
    feats_center = np.load(args.a1_dir + '/feats/Center_feats.dat.npy')
    feats_right = np.load(args.a1_dir + '/feats/Right_feats.dat.npy')
    feats_alt = np.load(args.a1_dir + '/feats/Alt_feats.dat.npy')
    id_left = (open(args.a1_dir + '/feats/Left_IDs.txt', 'r').read()).split('\n')
    id_center = (open(args.a1_dir + '/feats/Center_IDs.txt', 'r').read()).split('\n')
    id_right = (open(args.a1_dir + '/feats/Right_IDs.txt', 'r').read()).split('\n')
    id_alt = (open(args.a1_dir + '/feats/Alt_IDs.txt', 'r').read()).split('\n')

    # Create master dict of BGL
    BGL_path = args.a1_dir + '/Wordlists/BristolNorms+GilhoolyLogie.csv'
    BGL_dict = {}
    f = open(BGL_path, 'r')
    first_row = 1
    keys = []
    for row in f:
        if first_row:
            first_row = 0
            keys = row.strip()
            keys = keys.split(',')
        else:
            values = row.strip()
            values = values.split(',')
            if (values[1] != '' and values[1] not in BGL_dict):
                BGL_dict[values[1]] = {
                    keys[3]: values[3],
                    keys[4]: values[4],
                    keys[5]: values[5]
                }
    # Create master dict of War
    War_path = args.a1_dir + '/Wordlists/Ratings_Warriner_et_al.csv'
    War_dict = {}
    f = open(War_path, 'r')
    first_row = 1
    keys = []
    for row in f:
        if first_row:
            first_row = 0
            keys = row.strip()
            keys = keys.split(',')
        else:
            values = row.strip()
            values = values.split(',')
            if (values[1] != '' and values[1] not in War_dict):
                War_dict[values[1]] = {
                    keys[2]: values[2],
                    keys[5]: values[5],
                    keys[8]: values[8]
                }
    i = 0
    for sent in tqdm(data):
        # TODO: Use extract1 to find the first 29 features for each
        # data point. Add these to feats.
        feats[i][:29] = extract1(sent['body'], BGL_dict, War_dict)
        # TODO: Use extract2 to copy LIWC features (features 30-173)
        # into feats. (Note that these rely on each data point's class,
        # which is why we can't add them in extract1).
        feats[i][:174] = extract2(feats[i][:174], sent['cat'], sent['id'], feats_left, feats_center,
        feats_right, feats_alt, id_left, id_center, id_right, id_alt)
        i ++

    np.savez_compressed(args.output, feats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1")
    args = parser.parse_args()

    main(args)
