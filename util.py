from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import nltk
nltk.download('punkt') # NLTK提供的一個預訓練模型, 用於將文本分割成句子和單字
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt

















def calculate_rouge_bleu_and_pearson(summaries_list):
    rouge_scores_all = []
    bleu_scores_all = []
    
    # 對每個文本計算ROUGE與BLEU分數
    for key, truth_summary, generate_summary in summaries_list:
        rouge_scores = calculate_rouge_scores(truth_summary, generate_summary)
        print(f'rouge_scores: {rouge_scores}')
        bleu_score = calculate_bleu(truth_summary, generate_summary)
        
        # 將每個文本的分數添加到串列
        rouge_scores_all.append([rouge_scores[f'rouge{n}'].fmeasure if n != 5 else rouge_scores['rougeL'].fmeasure for n in range(1, 6)])
        bleu_scores_all.append(bleu_score)
    
    print(f'rouge_scores_all: {rouge_scores_all}')
    # 將list轉換為nump array以便計算相關係數
    rouge_scores_np = np.array(rouge_scores_all)
    bleu_scores_np = np.array(bleu_scores_all)
    print(f'rouge_scores_np[:, 0]: {rouge_scores_np[:, 0]}')
    print(f'bleu_scores_np: {bleu_scores_np}')
    
    # 計算pearson相關係數
    # rouge_scores_np[:, 0]表示每個文本rouge-1的fmeasure
    pearson_correlations = {
        'pearson_corr_rouge_1': pearsonr(rouge_scores_np[:, 0], bleu_scores_np)[0],
        'pearson_corr_rouge_2': pearsonr(rouge_scores_np[:, 1], bleu_scores_np)[0],
        'pearson_corr_rouge_3': pearsonr(rouge_scores_np[:, 2], bleu_scores_np)[0],
        'pearson_corr_rouge_4': pearsonr(rouge_scores_np[:, 3], bleu_scores_np)[0],
        'pearson_corr_rouge_L': pearsonr(rouge_scores_np[:, 4], bleu_scores_np)[0]
    }
    

    results = []
    pearson_corr = []
    for i, (key, _, _) in enumerate(summaries_list):
        results.append({
            'key': key,
            'rouge_scores': rouge_scores_all[i],
            'bleu_score': bleu_scores_all[i],
        })
    pearson_corr.append({'pearson_correlations': pearson_correlations})
    return results, pearson_corr 




def plot(results, pearson_corr):
    # Extracting data for plotting
    keys = [result['key'] for result in results]
    bleu_scores = [result['bleu_score'] for result in results]
    rouge_1_scores = [result['rouge_scores'][0] for result in results]
    rouge_2_scores = [result['rouge_scores'][1] for result in results]
    rouge_3_scores = [result['rouge_scores'][2] for result in results]
    rouge_4_scores = [result['rouge_scores'][3] for result in results]
    rouge_L_scores = [result['rouge_scores'][4] for result in results]


    # Number of bars (groups)
    n_bars = len(results)
    bar_width = 0.15
    index = np.arange(n_bars)

    # Creating the plot
    plt.figure(figsize=(15, 8))

    plt.bar(index, bleu_scores, bar_width, label='BLEU', color='b')
    plt.bar(index + bar_width, rouge_1_scores, bar_width, label='ROUGE-1', color='r')
    plt.bar(index + 2*bar_width, rouge_2_scores, bar_width, label='ROUGE-2', color='g')
    plt.bar(index + 3*bar_width, rouge_3_scores, bar_width, label='ROUGE-3', color='y')
    plt.bar(index + 4*bar_width, rouge_4_scores, bar_width, label='ROUGE-4', color='purple')
    plt.bar(index + 5*bar_width, rouge_L_scores, bar_width, label='ROUGE-L', color='brown')

    plt.xlabel('Document Key')
    plt.ylabel('Scores')
    plt.title('BLEU and ROUGE Scores for Each Document')
    plt.xticks(index + bar_width*2, keys)
    plt.legend()


    # Extracting Pearson correlation coefficients for a separate plot
    pearson_values = list(pearson_corr[0]['pearson_correlations'].values())
    pearson_keys = list(pearson_corr[0]['pearson_correlations'].keys())

    # Plot for Pearson correlation coefficients
    plt.figure(figsize=(10, 5))  # Create a new figure for Pearson correlations
    bars = plt.bar(pearson_keys, pearson_values, color='orange')
    plt.xlabel('ROUGE Types')
    plt.ylabel('Pearson Correlation Coefficient')
    plt.title('Pearson Correlation Coefficients for ROUGE Types')
    # Add the text on the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 3), ha='center', va='bottom')
    plt.show()







def pre_processing(truth, generated):
    reference = word_tokenize(truth.lower())
    candidate = word_tokenize(generated.lower())
    # Remove punctuation from tokenized words
    reference = [word for word in reference if word.isalpha()]
    candidate = [word for word in candidate if word.isalpha()]
    return  reference, candidate




# 用於計算BLEU分數的函數
def calculate_bleu(reference, candidate):
    smoothie = SmoothingFunction().method2
    reference, candidate = pre_processing(reference, candidate)
    score = sentence_bleu([reference], candidate, smoothing_function=smoothie)
    return score









def calculate_bleu_for_summaries(summaries_list):
    """
    為一系列文本的真實摘要和生成摘要計算 BLEU 分數。

    Args:
    summaries_list (list of tuples): 每個元組包含 (文本ID, 真實摘要, 生成摘要)。

    Returns:
    list of tuples: 每個元組包含 (文本ID, BLEU 分數)。
    """
    bleu_scores = []
    for key, truth_summary, generate_summary in summaries_list:
        score = calculate_bleu(truth_summary, generate_summary)
        bleu_scores.append((key, score))
    return bleu_scores















def calculate_rouge_scores(truth_summary, generate_summary):
    truth_summary_tokens, generate_summary_tokens = pre_processing(truth_summary, generate_summary)
    truth_summary = ' '.join(truth_summary_tokens)
    generate_summary = ' '.join(generate_summary_tokens)
    """
    計算給定摘要的 1-gram 至 4-gram 的 ROUGE 分數。

    Args:
    truth_summary (str): 真實摘要文本。
    generate_summary (str): 模型生成的摘要文本。

    Returns:
    dict: 各種 n-gram ROUGE 分數。
    """
    scores = {}
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3', 'rouge4', 'rougeL'], use_stemmer=False)

    # 為 1-gram 至 4-gram 計算 ROUGE 分數
    for n in range(1, 6):
        rouge_type = f'rouge{n}'
        if n == 5:
            rouge_type = 'rougeL'
        print(f'truth_summary: {truth_summary}')
        print(f'generate_summary: {generate_summary}')
        score = scorer.score(truth_summary, generate_summary)[rouge_type]
        scores[rouge_type] = score

    return scores

















def write_summaries_to_file(summary_dict, output_file_path):
    """
    函式用於將摘要信息寫入到指定的文件中。

    Args:
    summary_dict (dict): 包含真實摘要(truth_summary)和模型生成摘要(generate_summary)的字典。
    output_file_path (str): 輸出文件的路徑。
    """
    summaries_return = []
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for key, summaries in summary_dict.items():
            file.write(f"文本 {key}:\n")
            file.write(f"真實摘要: {summaries['truth_summary']}\n")
            file.write(f"模型生成摘要: {summaries['generate_summary']}\n")
            file.write("\n")  # 添加一個空行作為分隔
            summaries_return.append((key, summaries['truth_summary'], summaries['generate_summary']))
    return summaries_return


def extract_summaries_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    # 初始化兩個空列表來存儲真實摘要和模型生成摘要
    real_summaries = []
    generated_summaries = []

    # 讀取每一行並提取摘要
    for line in lines:
        if line.startswith("真實摘要"):
            # 提取真實摘要文本並刪除開頭的"–"字符，如果存在
            real_summary = line.split(": ", 1)[1].strip().lstrip("–").strip()
            real_summaries.append(real_summary)
        elif line.startswith("模型生成摘要"):
            # 提取模型生成摘要文本並刪除開頭的"–"字符，如果存在
            generated_summary = line.split(": ", 1)[1].strip().lstrip("–").strip()
            generated_summaries.append(generated_summary)
    return {i: (real_summaries[i], generated_summaries[i]) for i in range(len(real_summaries))}


def validate_summaries(file_path, summary_dict):
    summary_pairs = extract_summaries_from_file(file_path)

    # 驗證每一對摘要是否匹配
    for i in summary_dict.keys():
        truth_summary = summary_dict[i]['truth_summary']
        generate_summary = summary_dict[i]['generate_summary']
        real_summary = summary_pairs[i][0]
        generated_summary = summary_pairs[i][1]

        if truth_summary != real_summary or generate_summary != generated_summary:
            return False  # 如果有任何不匹配，返回False

    return True  # 所有摘要都匹配，返回True








# 修改函式以刪除摘要開頭的"–"字符

def extract_summaries_cleaned(file_path):
    """
    函式用於從指定文件中提取真實摘要和模型生成摘要，並以特定格式返回。
    
    Args:
    file_path (str): 要讀取的文件路徑。
    
    Returns:
    dict: 包含真實摘要(truth_summary)和模型生成摘要(generate_summary)的字典。
    """
    summary_pairs = extract_summaries_from_file(file_path)

    # 將摘要對應到指定的key
    summary_dict = {}
    for i, (truth_summary, generate_summary) in summary_pairs.items():
        summary_dict[i] = {"truth_summary": truth_summary, "generate_summary": generate_summary}
    
    return summary_dict




def main():
    file_path = './results.txt'
    # 調用函數並獲取含有key的摘要字典
    summaries_with_keys = extract_summaries_cleaned(file_path)
    # print(summaries_with_keys)   # 顯示摘要字典

    # 驗證用
    validation_result = validate_summaries(file_path, summaries_with_keys)
    print(validation_result)

    output_file_path = './summaries_output.txt'
    summaries = write_summaries_to_file(summaries_with_keys, output_file_path)
    # print(f'summaries: {summaries}')

    file_path = './summaries_output.txt'
    summaries_with_keys = extract_summaries_cleaned(file_path)

    # 選擇一個摘要進行 ROUGE 分數計算
    
    sample_key = 10
    truth_summary = summaries_with_keys[sample_key]['truth_summary']
    generate_summary = summaries_with_keys[sample_key]['generate_summary']
    # 計算並打印 ROUGE 分數
    rouge_scores = calculate_rouge_scores(truth_summary, generate_summary)
    print(rouge_scores)



    print(f'BLEU-4(smooth): {calculate_bleu_for_summaries(summaries)}')
    print(f'summaries: {summaries}')
    results, pearson_corr = calculate_rouge_bleu_and_pearson(summaries)
    print(f'results: {results}')
    print(f'pearson_corr: {pearson_corr}')
    plot(results, pearson_corr)






if __name__ == "__main__":
    main()