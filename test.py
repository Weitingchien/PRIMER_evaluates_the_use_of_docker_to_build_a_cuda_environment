import math
from rouge_score import rouge_scorer
import numpy as np
from scipy.stats import pearsonr
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
import nltk
nltk.download('punkt')



def main():

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    # generated_summary = "the us coast guard says it looking for a man who has cost the service about after responding to nearly of his fake distress calls the ap reports in a press release published friday the coast guard says the calls have originated from around the area of annapolis maryland each call involved the same male voice and used an emergency radio channel he been making the calls since july the two most recent calls were made on the night of july and the early morning of july the coast guard also says hoax calls distract rescuers from real emergencies putting both the public and the responding crews at risk"
    # gt_summary = "the us coast guard says it looking for a man who has cost the service about after responding to nearly of his fake distress calls reports the ap in a press release published friday the coast guard says the calls have originated from around the area of annapolis maryland each call involved the same male voice and used an emergency radio channel he been making the calls since july the two most recent calls were made on the night of july and the early morning of july a hoax call is a deadly and serious offense a coast guard rep tells which notes that such calls are a felony that carry six years in prison civil fine criminal fine and reimbursement to the coast guard calls like these not only put our crews at risk but they put the lives of the public at risk"
    generated_summary = "A fast brown fox leaped over the lazy dog"
    gt_summary = "The quick brown fox jumps over the lazy dog"
    print(f'generated_summary: {len(generated_summary)}')
    # 計算 ROUGE-L 分數
    scores = scorer.score(gt_summary, generated_summary)
    # 獲取 ROUGE-L 分數
    rouge_l_score = scores['rougeL']

    print("Precision:", rouge_l_score.precision)
    print("Recall:", rouge_l_score.recall)
    print("F-measure:", rouge_l_score.fmeasure)


if __name__ == "__main__":
    main()






