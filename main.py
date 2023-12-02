import torch
import random
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, LEDForConditionalGeneration




def main():
     # 檢查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')

    dataset=load_dataset('multi_news')
    # 獲取數據集的一個樣本
    sample = dataset['train'][0]  # 這裡假設你想查看訓練集中的第一個樣本
    print(f'sample: {sample}')
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
    # config = LongformerEncoderDecoderConfig.from_pretrained(PRIMER_path)
    # model = LongformerEncoderDecoderForConditionalGeneration.from_pretrained(PRIMER_path, config=config)
    model = LEDForConditionalGeneration.from_pretrained(PRIMER_path)

    print(f'model: {model}')
    PAD_TOKEN_ID = TOKENIZER.pad_token_id
    DOCSEP_TOKEN_ID = TOKENIZER.convert_tokens_to_ids("<doc-sep>")


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
            for doc in all_docs:
                input_ids.extend(
                    TOKENIZER.encode(
                        doc,
                        truncation=True,
                        max_length=4096 // len(all_docs),
                    )[1:-1]
                )
                input_ids.append(DOCSEP_TOKEN_ID)
            input_ids = (
                [TOKENIZER.bos_token_id]
                + input_ids
                + [TOKENIZER.eos_token_id]
            )
            # 確保序列長度符合模型要求
            while len(input_ids) % 1024 != 0:
                input_ids.append(PAD_TOKEN_ID)


            input_ids_all.append(torch.tensor(input_ids))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_all, batch_first=True, padding_value=PAD_TOKEN_ID
        )
        print(f'input_ids: {input_ids}')
        """
        max_token_id = max([max(ids) for ids in input_ids_all])
        if max_token_id >= TOKENIZER.vocab_size:
            print(f"Token ID {max_token_id} is out of range for the tokenizer vocabulary.")
        """

        return input_ids


    def batch_process(batch):
        input_ids=process_document(batch['document'])
        # get the input ids and attention masks together
        global_attention_mask = torch.zeros_like(input_ids).to(input_ids.device)
        # put global attention on <s> token

        global_attention_mask[:, 0] = 1
        global_attention_mask[input_ids == DOCSEP_TOKEN_ID] = 1
        generated_ids = model.generate(
            input_ids=input_ids,
            global_attention_mask=global_attention_mask,
            use_cache=True,
            max_length=1024,
            num_beams=5,
     )
        generated_str = TOKENIZER.batch_decode(
                generated_ids.tolist(), skip_special_tokens=True
            )
        result={}
        result['generated_summaries'] = generated_str
        result['gt_summaries']=batch['summary']
        return result



    data_idx = random.choices(range(len(dataset['test'])),k=10)
    dataset_small = dataset['test'].select(data_idx)
    result_small = dataset_small.map(batch_process, batched=True, batch_size=2)
    rouge = evaluate.load('rouge')
    results=rouge.compute(predictions=result_small["generated_summaries"], references=result_small["gt_summaries"])
    print(results)




if __name__ == "__main__":
    main()
