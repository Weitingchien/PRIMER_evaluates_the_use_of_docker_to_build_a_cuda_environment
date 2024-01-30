import torch
import random
import evaluate
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from datasets import load_dataset
from transformers import AutoTokenizer, LEDForConditionalGeneration
from rouge_score import rouge_scorer




def write_summaries_and_scores_to_file(result_small, file_name='results.txt'):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3', 'rouge4', 'rougeL', 'rougeLsum'], use_stemmer=True)

    with open(file_name, 'w', encoding='utf-8') as file:
        for i in range(len(result_small['document'])):
            original_article = result_small['document'][i]
            real_summary = result_small['gt_summaries'][i]
            generated_summary = result_small['generated_summaries'][i]

            # 計算BLEU和ROUGE分數
            gen_summary_words = word_tokenize(generated_summary)
            gt_summary_words = word_tokenize(real_summary)
            bleu_score = sentence_bleu([gt_summary_words], gen_summary_words)
            rouge_score = scorer.score(real_summary, generated_summary)

            # 寫入文件
            file.write(f"原始文章 {i+1}: {original_article}\n")
            file.write(f"真實摘要 {i+1}: {real_summary}\n")
            file.write(f"模型生成摘要 {i+1}: {generated_summary}\n")
            file.write(f"BLEU分數 {i+1}: {bleu_score}\n")
            file.write(f"ROUGE分數 {i+1}: {rouge_score}\n\n")












def main():
    torch.set_printoptions(threshold=10_000)
     # 檢查是否有可用的 GPU
    device = "cpu"
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f'device: {device}')

    dataset=load_dataset('multi_news')
    # 獲取數據集的一個樣本
    sample = dataset['train'][0]  # 這裡假設你想查看訓練集中的第一個樣本
    # print(f'sample: {sample}')
    # 解壓縮 PRIMER_multinews.tar.gz
    """
    file_path = '../../../Downloads/PRIMER_multinews.tar.gz'
    extraction_path = './PRIMER_pretrained'
    if not os.path.exists(extraction_path):
        os.makedirs(extraction_path)
    with tarfile.open(file_path, 'r:gz') as tar:
        # Extract all the contents into the directory
        tar.extractall(path=extraction_path)
    """


    PRIMER_path='./PRIMER_multinews'
    TOKENIZER = AutoTokenizer.from_pretrained(PRIMER_path)
    # 檢查詞彙大小
    print("Original tokenizer vocab size:", TOKENIZER.vocab_size)
    special_tokens_dict = {'additional_special_tokens': ['<doc-sep>']}
    TOKENIZER.add_special_tokens(special_tokens_dict)
    DOCSEP_TOKEN_ID = TOKENIZER.convert_tokens_to_ids("<doc-sep>")
    # config = LongformerEncoderDecoderConfig.from_pretrained(PRIMER_path)
    # model = LongformerEncoderDecoderForConditionalGeneration.from_pretrained(PRIMER_path, config=config)
    model = LEDForConditionalGeneration.from_pretrained(PRIMER_path).to(device)
    model.resize_token_embeddings(len(TOKENIZER))  # 模型需要調整以匹配新的TOKENIZER大小
    # 檢查嵌入層大小
    print("Model embeddings size:", model.get_input_embeddings().num_embeddings)

    print(f'model: {model}')
    PAD_TOKEN_ID = TOKENIZER.pad_token_id
    


    def process_document(documents):
        input_ids_all=[]
        for data in documents:
            all_docs = data.split("|||||")[:-1]
            for i, doc in enumerate(all_docs):
                doc = doc.replace("\n", " ")
                doc = " ".join(doc.split())
                all_docs[i] = doc

            #### concat with global attention on doc-sep
            input_ids = []
            for i, doc in enumerate(all_docs):
                input_ids.extend(
                    TOKENIZER.encode(
                        doc,
                        truncation=True,
                        max_length=4096 // len(all_docs),
                    )[1:-1]
                )
                input_ids.append(DOCSEP_TOKEN_ID)
                # print(f'all_docs: {len(all_docs)}')

            print(f'DOCSEP_TOKEN_ID: {DOCSEP_TOKEN_ID}')
                
            input_ids = (
                [TOKENIZER.bos_token_id]
                + input_ids
                + [TOKENIZER.eos_token_id]
            )
            input_ids_all.append(torch.tensor(input_ids).to(device))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_all, batch_first=True, padding_value=PAD_TOKEN_ID
        )
        # print(f'input_ids_all: {input_ids_all} len: {len(input_ids_all)}')

        
        max_token_id = max([max(ids) for ids in input_ids_all])
        if max_token_id >= TOKENIZER.vocab_size:
            print(f'TOKENIZER.vocab_size: {TOKENIZER.vocab_size}')
            print(f"Token ID {max_token_id} is out of range for the tokenizer vocabulary.")
        

        return input_ids


    def batch_process(batch):
        # print(f'batch: {batch}')
        input_ids=process_document(batch['document'])
        # get the input ids and attention masks together
        global_attention_mask = torch.zeros_like(input_ids).to(input_ids.device)
        # put global attention on <s> token
        # 每個序列的第一個token(<s>)的global attention設置為1, 表示模型應該關注這個token
        global_attention_mask[:, 0] = 1
        # 將所有包含特殊token`<doc-sep>`位置的global attention設置為1, 讓模型在文檔的分隔點上也給予global attention 
        global_attention_mask[input_ids == DOCSEP_TOKEN_ID] = 1
        # print(f'global_attention_mask: {global_attention_mask}')
        # 使用generate函式生成摘要, max_length指定生成摘要的最大長度
        generated_ids = model.generate(
            input_ids=input_ids,
            global_attention_mask=global_attention_mask,
            use_cache=True,
            max_length=1024,
            num_beams=5,
        )
        # print(f'generated_ids: {generated_ids}')
        #  將生成的generated_ids轉換為文本
        generated_str = TOKENIZER.batch_decode(
                generated_ids.tolist(), skip_special_tokens=True
            )
        result={}
        # 生成的摘要
        result['generated_summaries'] = generated_str
        # 真實的摘要(專業編輯人士所撰寫的摘要)
        result['gt_summaries']=batch['summary']
        # 印出原始文章、真實摘要和模型生成的摘要
        """
        for doc, real_summary, gen_summary in zip(batch['document'], batch['summary'], generated_str):
            print("原始文章：", doc)
            print("真實摘要：", real_summary)
            print("模型生成摘要：", gen_summary)
            print("\n")
        """


        return result


    """
    # 隨機選取10篇文章進行摘要生成
    data_idx = random.choices(range(len(dataset['test'])),k=10)
    dataset_small = dataset['test'].select(data_idx)
    result_small = dataset_small.map(batch_process, batched=True, batch_size=2)
    rouge = evaluate.load('rouge')
    results=rouge.compute(predictions=result_small["generated_summaries"], references=result_small["gt_summaries"])
    print(results)
    """

    # 篩選測試集的前11筆數據
    data_idx = list(range(11))
    dataset_small = dataset['test'].select(data_idx)
    result_small = dataset_small.map(batch_process, batched=True, batch_size=2)
    print(f'result_small: {result_small} len: {len(result_small)}')
    print(f'len result_small["generated_summaries"]: {len(result_small["generated_summaries"])}')
    print(f'len result_small["gt_summaries"]: {len(result_small["gt_summaries"])}')

    """ 計算這11筆的平均Rouge分數 """
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=result_small["generated_summaries"], references=result_small["gt_summaries"])
    # print(results)


    """ 每個樣本的Rouge分數 """
    """
    for i in range(len(result_small['generated_summaries'])):
        generated_summary = result_small['generated_summaries'][i]
        gt_summary = result_small['gt_summaries'][i]
        results = rouge.compute(predictions=[generated_summary], references=[gt_summary])
        print(f"第 {i+1} 的Rouge分數: {results}")
    """
    
    

    # 計算ROUGE與BLEU分數
    write_summaries_and_scores_to_file(result_small)

    index = 10
    if index < len(result_small['document']):
        original_article = result_small['document'][index]
        real_summary = result_small['gt_summaries'][index]
        generated_summary = result_small['generated_summaries'][index]

        print("第11篇原始文章:", original_article)
        print("第11篇真實摘要:", real_summary)
        print("第11篇模型生成摘要:", generated_summary)
    else:
        print(f"數據集中沒有第 {index + 1} 篇文章。")

    





if __name__ == "__main__":
    main()
