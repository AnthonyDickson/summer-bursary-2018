import sys
import argparse

import matplotlib.pyplot as plt
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS

def get_stopwords():
    stopwords = set(STOPWORDS)
    
#     stopwords.add('page')
    stopwords.add('paper')
#     stopwords.add('readings')
#     stopwords.add('research')
#     stopwords.add('summary')
#     stopwords.add('wikipedia')
    stopwords.add('________________')
    stopwords.add('https')
    stopwords.add('Anthony')
    stopwords.add('Dickson')

#     stopwords.add('look')
#     stopwords.add('possible')
#     stopwords.add('related')
#     stopwords.add('seemed')
#     stopwords.add('seems')
#     stopwords.add('understand')
#     stopwords.add('use')
#     stopwords.add('used')
#     stopwords.add('using')
    
#     stopwords.add('date')
#     stopwords.add('november')
#     stopwords.add('week')
#     stopwords.add('monday')
#     stopwords.add('tuesday')
#     stopwords.add('wednesday')
#     stopwords.add('thursday')
#     stopwords.add('friday')
    stopwords.add('Jan')
    stopwords.add('Feb')
    stopwords.add('Mar')
    stopwords.add('Apr')
    stopwords.add('May')
    stopwords.add('Jun')
    stopwords.add('Jul')
    stopwords.add('Aug')
    stopwords.add('Sep')
    stopwords.add('Oct')
    stopwords.add('Nov')
    stopwords.add('November')
    stopwords.add('Dec')
    
    return stopwords

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Generate a wordcloud from a txt fiile.')
    parser.add_argument('--infile', type=str, required = True,
                       help='the txt file from which to generate the wordcloud')
    parser.add_argument('--outfile', type=str,
                       help='the name of the destination file, including the file extension.')
    parser.add_argument('--width', type=int, default=1920,
                       help='the width of the output image')
    parser.add_argument('--height', type=int, default=1080,
                       help='the height of the output image')

    args = parser.parse_args()
    
    with open(args.infile, 'r') as f:
        data = ''

        for line in f:
            data += line
            
    out_filename = args.outfile if args.outfile is not None else args.infile + '-wordcloud.png'


    wordcloud = WordCloud(stopwords=get_stopwords(), 
                          width=args.width,
                          height=args.height, 
                          background_color='white',
                          random_state=42)

    wordcloud_out = wordcloud.generate(data)
    plt.imsave(out_filename, wordcloud_out)

