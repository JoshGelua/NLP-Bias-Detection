import sys
import argparse
import os
import json
import re
import spacy
import html


nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)


def preproc1(comment , steps=range(1, 6)):
    ''' This function pre-processes a single comment

    Parameters:
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step

    Returns:
        modComm : string, the modified comment
    '''
    modComm = comment
    if 1 in steps:  # replace newlines with spaces
        modComm = re.sub(r"\n{1,}", " ", modComm)
    if 2 in steps:  # unescape html
        modComm = html.unescape(modComm)
    if 3 in steps:  # remove URLs
        modComm = re.sub(r"(http|www)\S+", "", modComm)
    if 4 in steps:  # remove duplicate spaces
        modComm = " ".join(modComm.split())

    # TODO: get Spacy document for modComm
    if 5 in steps:
        spacy_doc = nlp(modComm)
        # TODO: use Spacy document for modComm to create a string.
        newline = ""
        for word in spacy_doc.sents:
            word_suffixed = []
            for suffix in word:
                if suffix.lemma_[0] != "-" or suffix.text[0] == "-":
                    word_suffixed.append(suffix.lemma_ + "/" + suffix.tag_)
                else:
                    word_suffixed.append(suffix.text + "/" + suffix.tag_)
            newline += " ".join(word_suffixed)
            newline += '\n'
        modComm = newline

    # Make sure to:
    #    * Insert "\n" between sentences.
    #    * Split tokens with spaces.
    #    * Write "/POS" after each token.

    return modComm


def main(args):
    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print( "Processing " + fullFile)
            data = json.load(open(fullFile))

            # TODO: select appropriate args.max lines
            sliced_index = args.ID[0] % len(data)

            for line in range(sliced_index, sliced_index + args.max):
                # circular indexing
                if line >= len(data):
                    line -= len(data)
                # read those lines with something like `j = json.loads(line)`
                j = json.loads(data[line])
                # choose to retain fields from those lines that are relevant to you
                processed = {
                #
                "id": j["id"],
                # add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...)
                "cat": file,
                # process the body field (j['body']) with preproc1(...) using default for `steps` argument
                # replace the 'body' field with the processed text
                "body": preproc1(j['body'])
                }
                # append the result to 'allOutput'
                allOutput.append(processed)

    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", type=int, help="The maximum number of comments to read from each file", default=10000)
    parser.add_argument("--a1_dir", help="The directory for A1. Should contain subdir data. Defaults to the directory for A1 on cdf.", default='/u/cs401/A1')

    args = parser.parse_args()

    if (args.max > 200272):
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)

    indir = os.path.join(args.a1_dir, 'data')
    main(args)
